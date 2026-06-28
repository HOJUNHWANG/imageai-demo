const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api";

export type ProfileId = "quality" | "balanced" | "fast";

export type ModelProfile = {
  id: ProfileId;
  label: string;
  model_id: string;
  family: string;
  description: string;
  steps: number;
  guidance: number;
  true_cfg: number;
  long_side: number;
  max_pixels: number;
  transformer_id: string | null;
  lora_id: string | null;
  lora_weight: string | null;
  prequantized: boolean;
  gated: boolean;
  content_tuning: string;
};

export type StudioConfig = {
  default_profile: ProfileId;
  profiles: Record<"generate" | "edit", Record<ProfileId, ModelProfile>>;
  safety_checker: boolean;
  attention_backend: string;
  note: string;
};

export type JobStatus = {
  active: boolean;
  task: string;
  stage: string;
  message: string;
  step: number;
  total: number;
  stage_progress: number | null;
  overall_progress: number;
  indeterminate: boolean;
  eta_seconds: number | null;
  stage_elapsed: number;
  elapsed: number;
};

export type StudioStatus = {
  job: JobStatus;
  model: {
    loaded: boolean;
    kind: "generate" | "edit" | null;
    profile: ProfileId | null;
    model: string | null;
    four_bit: boolean;
    text_encoder_four_bit: boolean;
    cpu_offload: boolean;
    attention_backend: string;
  };
  hardware: {
    device: string;
    gpu_name: string | null;
    vram_total_gb: number;
    vram_allocated_gb: number;
    ram_total_gb: number;
    ram_used_gb: number;
  };
};

export type ImageResult = {
  image: string;
  seed: number;
  elapsed: number;
  width: number;
  height: number;
  masked?: boolean;
  profile: ProfileId;
  model: string;
  warm_model: boolean;
  timings: {
    load: number;
    inference: number;
    postprocess: number;
    total: number;
  };
};

async function responseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || body.error || message;
    } catch { /* response had no JSON body */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function getStatus(): Promise<StudioStatus> {
  return responseJson(await fetch(`${API_BASE}/status`, { cache: "no-store" }));
}

export async function getConfig(): Promise<StudioConfig> {
  return responseJson(await fetch(`${API_BASE}/config`, { cache: "no-store" }));
}

export async function generateImage(input: {
  prompt: string;
  width: number;
  height: number;
  profile: ProfileId;
  seed: number;
}): Promise<ImageResult> {
  return responseJson(await fetch(`${API_BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  }));
}

export async function editImage(form: FormData): Promise<ImageResult> {
  return responseJson(await fetch(`${API_BASE}/edit`, { method: "POST", body: form }));
}

export async function cancelJob(): Promise<void> {
  await responseJson(await fetch(`${API_BASE}/cancel`, { method: "POST" }));
}

export async function unloadModel(): Promise<void> {
  await responseJson(await fetch(`${API_BASE}/unload`, { method: "POST" }));
}
