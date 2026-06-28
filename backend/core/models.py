"""One-profile-at-a-time model loader tuned for a 12 GB RTX 3080 Ti."""
from __future__ import annotations

import logging
import os
import threading
import time

import torch

from .config import (
    ATTENTION_BACKEND,
    CPU_OFFLOAD,
    DEVICE,
    DTYPE,
    ENABLE_4BIT,
    HF_TOKEN,
    LOCAL_FILES_ONLY,
    QUANTIZE_TEXT_ENCODERS,
    ModelProfile,
    get_profile,
)
from .runtime import JOB, clear_memory

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self) -> None:
        self._pipe = None
        self._kind: str | None = None
        self._profile: ModelProfile | None = None
        self._lock = threading.RLock()
        self._loaded_at = 0.0
        self._attention_backend = "native"
        self._load_stage = "loading"

    @property
    def status(self) -> dict:
        return {
            "loaded": self._pipe is not None,
            "kind": self._kind,
            "profile": self._profile.id if self._profile else None,
            "model": self._profile.model_id if self._profile else None,
            "four_bit": ENABLE_4BIT,
            "text_encoder_four_bit": ENABLE_4BIT and QUANTIZE_TEXT_ENCODERS,
            "cpu_offload": CPU_OFFLOAD,
            "attention_backend": self._attention_backend,
            "loaded_at": self._loaded_at,
        }

    def unload(self) -> None:
        with self._lock:
            self._pipe = None
            self._kind = None
            self._profile = None
            self._loaded_at = 0.0
            self._attention_backend = "native"
            clear_memory()

    def get(self, kind: str, profile_id: str):
        profile = get_profile(kind, profile_id)
        with self._lock:
            if self._pipe is not None and self._kind == kind and self._profile == profile:
                return self._pipe, 0.0, True

            self.unload()
            self._load_stage = "loading" if self._profile_cached(profile) else "downloading"
            message = (
                f"Loading cached {profile.label} model"
                if self._load_stage == "loading"
                else f"Downloading {profile.label} model for first use"
            )
            JOB.update(self._load_stage, message, stage_progress=0.02 if self._load_stage == "loading" else None)
            started = time.perf_counter()
            try:
                self._pipe = self._load_generate(profile) if kind == "generate" else self._load_edit(profile)
            except Exception:
                self._pipe = None
                clear_memory()
                raise
            self._kind = kind
            self._profile = profile
            self._loaded_at = time.time()
            elapsed = time.perf_counter() - started
            logger.info("Loaded %s/%s in %.1fs", kind, profile.id, elapsed)
            return self._pipe, elapsed, False

    @staticmethod
    def _file_cached(model_id: str, filename: str) -> bool:
        if os.path.isdir(model_id):
            return os.path.isfile(os.path.join(model_id, filename))
        try:
            from huggingface_hub import try_to_load_from_cache

            return isinstance(try_to_load_from_cache(model_id, filename), str)
        except Exception:
            return False

    def _profile_cached(self, profile: ModelProfile) -> bool:
        required = [(profile.model_id, "model_index.json")]
        if profile.transformer_id:
            required.append((profile.transformer_id, "config.json"))
        if profile.lora_id and profile.lora_weight:
            required.append((profile.lora_id, profile.lora_weight))
        return all(self._file_cached(model_id, filename) for model_id, filename in required)

    def _load_update(self, message: str, progress: float) -> None:
        is_download = self._load_stage == "downloading"
        prefix = "Downloading / " if is_download else ""
        JOB.update(self._load_stage, prefix + message, stage_progress=None if is_download else progress)

    @staticmethod
    def _common_kwargs() -> dict:
        return {
            "torch_dtype": DTYPE,
            "low_cpu_mem_usage": True,
            "local_files_only": LOCAL_FILES_ONLY,
            "token": HF_TOKEN,
        }

    @staticmethod
    def _diffusers_quant_config():
        if not (ENABLE_4BIT and DEVICE == "cuda"):
            return None
        from diffusers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    @staticmethod
    def _transformers_quant_config():
        if not (ENABLE_4BIT and QUANTIZE_TEXT_ENCODERS and DEVICE == "cuda"):
            return None
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def _load_text_encoder(self, cls, model_id: str, subfolder: str):
        quant = self._transformers_quant_config()
        if quant is None:
            return None
        return cls.from_pretrained(
            model_id,
            subfolder=subfolder,
            quantization_config=quant,
            **self._common_kwargs(),
        )

    def _finish_pipeline(self, pipe):
        JOB.update("loading", "Configuring attention and memory offload", stage_progress=0.88)
        if DEVICE == "cuda" and ATTENTION_BACKEND:
            try:
                pipe.transformer.set_attention_backend(ATTENTION_BACKEND)
                self._attention_backend = ATTENTION_BACKEND
                logger.info("Using %s attention", ATTENTION_BACKEND)
            except Exception as exc:
                self._attention_backend = "native"
                logger.warning("Could not enable %s attention; using native SDPA: %s", ATTENTION_BACKEND, exc)
        if DEVICE == "cuda":
            if CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        else:
            pipe.to("cpu")

        try:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        except (AttributeError, NotImplementedError):
            pass
        JOB.update("loading", "Model ready", stage_progress=1.0)
        return pipe

    def _load_generate(self, profile: ModelProfile):
        if profile.family == "flux":
            return self._load_flux(profile)
        return self._load_zimage(profile)

    def _load_zimage(self, profile: ModelProfile):
        from diffusers import ZImagePipeline, ZImageTransformer2DModel
        from transformers import AutoModel

        kwargs = self._common_kwargs()
        self._load_update("preparing text encoder", 0.12)
        text_encoder = self._load_text_encoder(AutoModel, profile.model_id, "text_encoder")
        if text_encoder is not None:
            kwargs["text_encoder"] = text_encoder

        # The Balanced repository already contains an NF4 transformer. Re-quantizing
        # it costs minutes and extra peak RAM, so load it as-is.
        quant = self._diffusers_quant_config()
        if quant is not None and not profile.prequantized:
            self._load_update("quantizing image transformer", 0.38)
            kwargs["transformer"] = ZImageTransformer2DModel.from_pretrained(
                profile.transformer_id or profile.model_id,
                subfolder=profile.transformer_subfolder,
                quantization_config=quant,
                **self._common_kwargs(),
            )
        self._load_update("assembling Z-Image pipeline", 0.68)
        pipe = ZImagePipeline.from_pretrained(profile.model_id, **kwargs)
        return self._finish_pipeline(pipe)

    def _load_flux(self, profile: ModelProfile):
        from diffusers import FluxPipeline, FluxTransformer2DModel
        from transformers import T5EncoderModel

        kwargs = self._common_kwargs()
        self._load_update("preparing T5 text encoder", 0.12)
        text_encoder_2 = self._load_text_encoder(T5EncoderModel, profile.model_id, "text_encoder_2")
        if text_encoder_2 is not None:
            kwargs["text_encoder_2"] = text_encoder_2

        quant = self._diffusers_quant_config()
        if quant is not None:
            self._load_update("quantizing FLUX transformer", 0.38)
            kwargs["transformer"] = FluxTransformer2DModel.from_pretrained(
                profile.transformer_id or profile.model_id,
                subfolder=profile.transformer_subfolder,
                quantization_config=quant,
                **self._common_kwargs(),
            )
        self._load_update("assembling FLUX pipeline", 0.68)
        pipe = FluxPipeline.from_pretrained(profile.model_id, **kwargs)
        return self._finish_pipeline(pipe)

    def _load_edit(self, profile: ModelProfile):
        from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
        from transformers import Qwen2_5_VLForConditionalGeneration

        kwargs = self._common_kwargs()
        self._load_update("preparing vision-language encoder", 0.12)
        text_encoder = self._load_text_encoder(
            Qwen2_5_VLForConditionalGeneration,
            profile.model_id,
            "text_encoder",
        )
        if text_encoder is not None:
            kwargs["text_encoder"] = text_encoder

        quant = self._diffusers_quant_config()
        if quant is not None:
            self._load_update("quantizing edit transformer", 0.38)
            transformer_kwargs = self._common_kwargs()
            if profile.transformer_subfolder is not None:
                transformer_kwargs["subfolder"] = profile.transformer_subfolder
            kwargs["transformer"] = QwenImageTransformer2DModel.from_pretrained(
                profile.transformer_id or profile.model_id,
                quantization_config=quant,
                **transformer_kwargs,
            )

        self._load_update("assembling Qwen edit pipeline", 0.66)
        pipe = QwenImageEditPlusPipeline.from_pretrained(profile.model_id, **kwargs)
        if profile.lora_id:
            self._load_update(f"attaching {profile.label} acceleration adapter", 0.8)
            pipe.load_lora_weights(
                profile.lora_id,
                weight_name=profile.lora_weight,
                local_files_only=LOCAL_FILES_ONLY,
                token=HF_TOKEN,
            )
        return self._finish_pipeline(pipe)


MODELS = ModelManager()
