/**
 * API client for ImageAI Studio backend.
 */
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api";

async function checkResponse(res: Response): Promise<Response> {
    if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try { const body = await res.json(); msg = body.detail || body.error || msg; } catch { }
        throw new Error(msg);
    }
    return res;
}

export interface ProgressInfo {
    active: boolean;
    task: string;
    step: number;
    total: number;
    status: string;
    message: string;
    elapsed: number;
}

export interface VramInfo {
    gpu_name: string;
    allocated_gb: number;
    reserved_gb: number;
    total_gb: number;
}

export interface GenerateRequest {
    prompt: string;
    width: number;
    height: number;
    steps: number;
    seed: number;
}

export interface GenerateResponse {
    image?: string;
    seed?: number;
    elapsed?: number;
    status: string;
    error?: string;
    vram?: VramInfo;
}

export interface EditResponse {
    image?: string;
    seed?: number;
    elapsed?: number;
    prompt_used?: string;
    status: string;
    error?: string;
    vram?: VramInfo;
}

export interface MaskResponse {
    masks?: string[];
    count?: number;
    status: string;
    error?: string;
}

// ─── Generate ───
export async function generateImage(req: GenerateRequest): Promise<GenerateResponse> {
    const res = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
    });
    return (await checkResponse(res)).json();
}

// ─── Edit ───
export async function editImage(formData: FormData): Promise<EditResponse> {
    const res = await fetch(`${API_BASE}/edit`, {
        method: "POST",
        body: formData,
    });
    return (await checkResponse(res)).json();
}

// ─── Mask ───
export async function autoMask(formData: FormData): Promise<MaskResponse> {
    const res = await fetch(`${API_BASE}/mask/auto`, {
        method: "POST",
        body: formData,
    });
    return (await checkResponse(res)).json();
}

export interface ClickMaskResponse {
    masks?: string[];
    scores?: number[];
    labels?: string[];
    selected?: number;
    status: string;
    error?: string;
}

export async function clickMask(formData: FormData): Promise<ClickMaskResponse> {
    const res = await fetch(`${API_BASE}/mask/click`, {
        method: "POST",
        body: formData,
    });
    return (await checkResponse(res)).json();
}

// ─── Upload ───
export async function uploadImage(formData: FormData) {
    const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
    });
    return (await checkResponse(res)).json();
}

// ─── System ───
export async function getProgress(): Promise<ProgressInfo> {
    const res = await fetch(`${API_BASE}/progress`);
    return (await checkResponse(res)).json();
}

export async function cancelInference(): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/cancel`, { method: "POST" });
    return (await checkResponse(res)).json();
}

export async function getVram(): Promise<VramInfo> {
    const res = await fetch(`${API_BASE}/vram`, { signal: AbortSignal.timeout(2000) });
    return (await checkResponse(res)).json();
}

export async function clearSoft() {
    const res = await fetch(`${API_BASE}/clear/soft`, { method: "POST" });
    return (await checkResponse(res)).json();
}

export async function clearHard() {
    const res = await fetch(`${API_BASE}/clear/hard`, { method: "POST" });
    return (await checkResponse(res)).json();
}

export async function clearAux() {
    const res = await fetch(`${API_BASE}/clear/aux`, { method: "POST" });
    return (await checkResponse(res)).json();
}

// ─── Enrich ───
export async function enrichPreview(prompt: string, negative: string, autoEnrich: boolean, preset: string = "general"): Promise<{ positive: string; negative: string }> {
    const res = await fetch(`${API_BASE}/enrich`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, negative, auto_enrich: autoEnrich, preset }),
    });
    return (await checkResponse(res)).json();
}

export interface PresetInfo {
    id: string;
    label: string;
    description: string;
    strength: number;
    guidance: number;
    steps: number;
}

export async function getPresets(): Promise<{ presets: PresetInfo[] }> {
    const res = await fetch(`${API_BASE}/presets`);
    return (await checkResponse(res)).json();
}

export async function getGenPresets(): Promise<{ presets: PresetInfo[] }> {
    const res = await fetch(`${API_BASE}/presets/gen`);
    return (await checkResponse(res)).json();
}

export async function enrichFlux(prompt: string, autoEnrich: boolean, preset: string = "general"): Promise<{ positive: string; negative: string }> {
    const res = await fetch(`${API_BASE}/enrich/flux`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, auto_enrich: autoEnrich, preset }),
    });
    return (await checkResponse(res)).json();
}
