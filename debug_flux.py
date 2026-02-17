
import os
import torch
from diffusers import FluxPipeline
import gc

# Force flush
import sys
sys.stdout.reconfigure(line_buffering=True)

print("1. Starting Debug Script...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"2. Device: {DEVICE}")

try:
    print("3. Loading FLUX Pipeline... (this may take a while if downloading)")
    dtype = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=dtype
    )
    print("4. Pipeline Loaded Successfully!")
    
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
        print("5. CPU Offload Enabled")

    print("6. Generating Test Image...")
    image = pipe(
        prompt="A cat holding a sign that says DEBUG",
        width=512,
        height=512,
        num_inference_steps=4,
        guidance_scale=0.0
    ).images[0]
    
    print("7. Image Generated!")
    image.save("debug_flux_output.png")
    print("8. Image Saved to debug_flux_output.png")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
