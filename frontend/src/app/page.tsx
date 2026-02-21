"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  generateImage,
  editImage,
  autoMask,
  clickMask,
  getProgress,
  getVram,
  enrichPreview,
  enrichFlux,
  cancelInference,
  getPresets,
  getGenPresets,
  type GenerateResponse,
  type EditResponse,
  type PresetInfo,
  type ProgressInfo,
  type VramInfo,
} from "@/lib/api";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"welcome" | "generate" | "edit">("welcome");
  const [sentImage, setSentImage] = useState<string | null>(null);
  const [vram, setVram] = useState<VramInfo | null>(null);

  const handleSendToEditor = useCallback((imageBase64: string) => {
    setSentImage(imageBase64);
    setActiveTab("edit");
  }, []);
  const handleClearSent = useCallback(() => setSentImage(null), []);

  useEffect(() => {
    const poll = () => { getVram().then(setVram).catch(() => { }); };
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      <nav style={{ width: 220, height: "100vh", position: "fixed", top: 0, left: 0, background: "var(--bg-secondary)", borderRight: "1px solid var(--border)", padding: "24px 16px", display: "flex", flexDirection: "column", gap: 8, zIndex: 100, overflowY: "auto" }}>
        <div style={{ fontSize: "1.15rem", fontWeight: 700, marginBottom: 20, color: "var(--accent)", cursor: "pointer" }} onClick={() => setActiveTab("welcome")}>✦ ImageAI Studio</div>
        <SidebarBtn icon="🏠" label="Home" active={activeTab === "welcome"} onClick={() => setActiveTab("welcome")} />
        <SidebarBtn icon="🎨" label="Generate" active={activeTab === "generate"} onClick={() => setActiveTab("generate")} />
        <SidebarBtn icon="✏️" label="Edit" active={activeTab === "edit"} onClick={() => setActiveTab("edit")} />
        <div style={{ flex: 1 }} />
        {vram && vram.gpu_name !== "(CPU mode)" && (
          <div style={{ background: "var(--bg-tertiary)", borderRadius: 10, padding: "10px 12px", border: "1px solid var(--border)", fontSize: "0.75rem", marginBottom: 8 }}>
            <div style={{ color: "var(--text-secondary)", fontWeight: 600, marginBottom: 6 }}>🖥️ {vram.gpu_name.replace("NVIDIA GeForce ", "")}</div>
            <div style={{ background: "var(--bg-primary)", borderRadius: 4, height: 6, overflow: "hidden", marginBottom: 4 }}>
              <div style={{ height: "100%", borderRadius: 4, width: `${(vram.allocated_gb / vram.total_gb) * 100}%`, background: vram.allocated_gb / vram.total_gb > 0.85 ? "var(--error)" : "var(--accent-gradient)", transition: "width 0.5s ease" }} />
            </div>
            <div style={{ color: "var(--text-muted)" }}>{vram.allocated_gb} / {vram.total_gb} GB</div>
          </div>
        )}
        <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", padding: "0 4px", paddingBottom: 48 }}>SDXL Inpaint + FLUX · Local GPU</div>
      </nav>
      <main style={{ flex: 1, padding: 32, overflowY: "auto", marginLeft: 220 }}>
        <div style={{ display: activeTab === "welcome" ? "block" : "none" }}><WelcomePage onNavigate={setActiveTab} /></div>
        <div style={{ display: activeTab === "generate" ? "block" : "none" }}><GeneratePage onSendToEditor={handleSendToEditor} /></div>
        <div style={{ display: activeTab === "edit" ? "block" : "none" }}><EditPage sentImage={sentImage} onClearSent={handleClearSent} /></div>
      </main>
    </div>
  );
}

function SidebarBtn({ icon, label, active, onClick }: { icon: string; label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 12px", borderRadius: 10, border: "none", background: active ? "var(--accent-glow)" : "transparent", color: active ? "var(--accent)" : "var(--text-secondary)", fontWeight: active ? 600 : 400, fontSize: "0.88rem", cursor: "pointer", transition: "all 0.2s", textAlign: "left", width: "100%" }}>
      <span style={{ fontSize: "1rem" }}>{icon}</span>{label}
    </button>
  );
}

function ProgressBar({ task, onCancel }: { task: string; onCancel: () => void }) {
  const [progress, setProgress] = useState<ProgressInfo | null>(null);
  useEffect(() => {
    const id = setInterval(async () => {
      try { const p = await getProgress(); if (p.task === task && (p.active || p.status === "done" || p.status === "error")) setProgress(p); } catch { }
    }, 400);
    return () => clearInterval(id);
  }, [task]);
  if (!progress || progress.status === "idle") return null;
  const pct = progress.total > 0 ? Math.round((progress.step / progress.total) * 100) : 0;
  const statusColor = progress.status === "error" ? "var(--error)" : progress.status === "done" ? "var(--success)" : "var(--accent)";
  const isRunning = progress.active === true;
  return (
    <div className="fade-in" style={{ background: "var(--bg-tertiary)", borderRadius: 12, padding: "12px 16px", border: "1px solid var(--border)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: "0.8rem", alignItems: "center" }}>
        <span style={{ color: statusColor, fontWeight: 500 }}>
          {progress.status === "loading_model" && `⏳ ${progress.message || 'Loading model...'}`}
          {progress.status === "preprocessing" && `🔧 ${progress.message || 'Preprocessing...'}`}
          {progress.status === "running" && `🚀 ${progress.message}`}
          {progress.status === "encoding" && `📦 ${progress.message || 'Encoding...'}`}
          {progress.status === "done" && `✅ ${progress.message}`}
          {progress.status === "error" && `❌ ${progress.message}`}
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: "var(--text-muted)" }}>{progress.elapsed}s</span>
          {isRunning && (
            <button onClick={onCancel} style={{
              background: "var(--error)", color: "white", border: "none", borderRadius: 6,
              padding: "3px 10px", fontSize: "0.72rem", cursor: "pointer", fontWeight: 600,
            }}>✕ Cancel</button>
          )}
        </div>
      </div>
      <div className="progress-bar-wrap">
        <div
          className={`progress-bar-fill ${progress.status === "loading_model" || progress.status === "preprocessing" ? "indeterminate" : ""}`}
          style={{ width: progress.status === "done" ? "100%" : progress.status === "loading_model" ? "100%" : progress.status === "preprocessing" ? "100%" : progress.status === "encoding" ? "95%" : `${Math.max(pct, 3)}%` }}
        />
      </div>
    </div>
  );
}

function EnrichedPreview({ label, text }: { label: string; text: string }) {
  if (!text) return null;
  return (
    <div style={{ background: "var(--bg-tertiary)", borderRadius: 8, padding: "8px 12px", border: "1px solid var(--border)", fontSize: "0.75rem" }}>
      <div style={{ color: "var(--accent)", fontWeight: 600, marginBottom: 4, fontSize: "0.7rem" }}>{label}</div>
      <div style={{ color: "var(--text-secondary)", lineHeight: 1.4, wordBreak: "break-word", maxHeight: 80, overflowY: "auto" }}>{text}</div>
    </div>
  );
}

// ═══════════════════════════════════════════
// WELCOME PAGE
// ═══════════════════════════════════════════
function WelcomePage({ onNavigate }: { onNavigate: (tab: "generate" | "edit") => void }) {
  return (
    <div className="fade-in" style={{ maxWidth: 800, margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: 48, marginTop: 40 }}>
        <h1 style={{ fontSize: "2.2rem", fontWeight: 700, marginBottom: 8, background: "var(--accent-gradient)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Welcome to ImageAI Studio</h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1.05rem", maxWidth: 500, margin: "0 auto" }}>Create and edit images with AI — powered by FLUX and SDXL on your local GPU.</p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 40 }}>
        <div className="glass-card" style={{ padding: 28, cursor: "pointer" }} onClick={() => onNavigate("generate")}>
          <div style={{ fontSize: "2rem", marginBottom: 12 }}>🎨</div>
          <h3 style={{ fontSize: "1.1rem", fontWeight: 600, marginBottom: 8 }}>Generate</h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", lineHeight: 1.5 }}>Create photorealistic images from text using FLUX Schnell with 4-bit quantization.</p>
        </div>
        <div className="glass-card" style={{ padding: 28, cursor: "pointer" }} onClick={() => onNavigate("edit")}>
          <div style={{ fontSize: "2rem", marginBottom: 12 }}>✏️</div>
          <h3 style={{ fontSize: "1.1rem", fontWeight: 600, marginBottom: 8 }}>Edit</h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", lineHeight: 1.5 }}>Modify images with AI inpainting — click to select regions, auto-mask, prompt enrichment.</p>
        </div>
      </div>
      <div className="glass-card" style={{ padding: 28 }}>
        <h3 style={{ fontSize: "1.1rem", fontWeight: 600, marginBottom: 16 }}>📖 How to Use</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Step n={1} title="Generate an Image" desc="Go to Generate → type a description → click Generate." />
          <Step n={2} title="Send to Editor" desc="Click 'Send to Editor →' to transfer to the Edit workspace." />
          <Step n={3} title="Create a Mask" desc="Click on the image to select a region (SAM AI), or use Auto Mask for predefined areas." />
          <Step n={4} title="Describe Your Edit" desc="Write what you want — 'red leather jacket', 'blonde hair' → click Apply." />
          <Step n={5} title="Cancel Anytime" desc="Press the red Cancel button during generation/editing to stop immediately." />
        </div>
      </div>
      <div style={{ textAlign: "center", marginTop: 32 }}>
        <button className="btn-primary" onClick={() => onNavigate("generate")} style={{ display: "inline-flex" }}>🚀 Start Creating</button>
      </div>
    </div>
  );
}

function Step({ n, title, desc }: { n: number; title: string; desc: string }) {
  return (
    <div style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
      <div style={{ width: 28, height: 28, borderRadius: "50%", background: "var(--accent-glow)", color: "var(--accent)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.8rem", fontWeight: 700, flexShrink: 0 }}>{n}</div>
      <div><div style={{ fontWeight: 600, fontSize: "0.9rem", marginBottom: 2 }}>{title}</div><div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.45 }}>{desc}</div></div>
    </div>
  );
}

// ═══════════════════════════════════════════
// GENERATE PAGE
// ═══════════════════════════════════════════
function GeneratePage({ onSendToEditor }: { onSendToEditor: (img: string) => void }) {
  const [prompt, setPrompt] = useState("");
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [steps, setSteps] = useState(4);
  const [seed, setSeed] = useState(-1);
  const [autoEnrich, setAutoEnrich] = useState(true);
  const [enrichedPrompt, setEnrichedPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);

  // New state for generation presets
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState("general");

  useEffect(() => {
    getGenPresets().then((res) => {
      if (res.presets) setPresets(res.presets);
    }).catch(console.error);
  }, []);

  useEffect(() => {
    if (!prompt.trim()) { setEnrichedPrompt(""); return; }
    const t = setTimeout(async () => {
      try { const r = await enrichFlux(prompt, autoEnrich, selectedPresetId); setEnrichedPrompt(r.positive); } catch { }
    }, 300);
    return () => clearTimeout(t);
  }, [prompt, autoEnrich, selectedPresetId]);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setLoading(true); setResult(null);
    try {
      const res = await generateImage({ prompt: autoEnrich ? enrichedPrompt || prompt : prompt, width, height, steps, seed });
      setResult(res);
    } catch (e: any) { setResult({ status: "error", error: e.message }); }
    setLoading(false);
  };

  const handleCancel = async () => {
    try { await cancelInference(); } catch { }
  };

  return (
    <div className="fade-in">
      <h1 style={{ fontSize: "1.5rem", fontWeight: 700, marginBottom: 4 }}>🎨 Generate</h1>
      <p style={{ color: "var(--text-secondary)", marginBottom: 24, fontSize: "0.9rem" }}>Create photorealistic images from text using FLUX</p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.4fr", gap: 24 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <textarea className="input-field" rows={4} placeholder={"Describe the image...\ne.g., A woman in a red dress walking through a sunlit garden"} value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleGenerate(); } }} />
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.85rem", color: "var(--text-secondary)", cursor: "pointer" }}>
              <input type="checkbox" checked={autoEnrich} onChange={(e) => setAutoEnrich(e.target.checked)} /> ✨ Photo-realistic enrich
            </label>
            {autoEnrich && presets.length > 0 && (
              <select
                className="input-field"
                style={{ padding: "4px 8px", fontSize: "0.8rem", width: "180px" }}
                value={selectedPresetId}
                onChange={(e) => setSelectedPresetId(e.target.value)}
              >
                {presets.map(p => (
                  <option key={p.id} value={p.id}>{p.label}</option>
                ))}
              </select>
            )}
          </div>
          {enrichedPrompt && <EnrichedPreview label="📝 Final Prompt (sent to FLUX)" text={enrichedPrompt} />}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <Slider label="Width" value={width} min={256} max={1536} step={64} onChange={setWidth} />
            <Slider label="Height" value={height} min={256} max={1536} step={64} onChange={setHeight} />
            <Slider label="Steps" value={steps} min={1} max={12} step={1} onChange={setSteps} />
            <div><label style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>Seed</label><input type="number" className="input-field" value={seed} onChange={(e) => setSeed(Number(e.target.value))} style={{ padding: "8px 12px" }} /></div>
          </div>
          {loading ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <button className="btn-primary" disabled style={{ opacity: 0.6 }}><span className="spinner" /> Generating...</button>
              <ProgressBar task="generate" onCancel={handleCancel} />
            </div>
          ) : (
            <button className="btn-primary" onClick={handleGenerate} disabled={!prompt.trim()}>🚀 Generate</button>
          )}
          {result && !loading && (
            <div className="fade-in">
              {result.status === "success" ? (<><div className="status-pill success">✓ Done in {result.elapsed}s · Seed: {result.seed}</div><button className="btn-secondary" style={{ marginTop: 8, width: "100%" }} onClick={() => result.image && onSendToEditor(result.image)}>📤 Send to Editor →</button></>) : result.status === "cancelled" ? (<div className="status-pill" style={{ background: "rgba(255,165,0,0.15)", color: "orange" }}>⚠️ Cancelled</div>) : (<div className="status-pill error">✗ {result.error}</div>)}
            </div>
          )}
        </div>
        <div className={`image-display ${loading ? "loading" : ""}`} style={{ minHeight: 500 }}>
          {result?.image ? (<img src={`data:image/jpeg;base64,${result.image}`} alt="Generated" className="scale-pop" />) : (<div style={{ color: "var(--text-muted)", textAlign: "center" }}><div style={{ fontSize: "3rem", marginBottom: 12, opacity: 0.3 }}>🖼️</div><div>Your generated image will appear here</div></div>)}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════
// EDIT PAGE — Source+Overlay | Result layout
// ═══════════════════════════════════════════
function EditPage({ sentImage, onClearSent }: { sentImage: string | null; onClearSent: () => void }) {
  const [workingImage, setWorkingImage] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [maskMode, setMaskMode] = useState<"click" | "auto">("click");
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [autoEnrich, setAutoEnrich] = useState(true);
  const [maskTarget, setMaskTarget] = useState("top");
  const [strength, setStrength] = useState(0.55);
  const [editSteps, setEditSteps] = useState(28);
  const [guidance, setGuidance] = useState(7.0);
  const [seed, setSeed] = useState(-1);
  const [maskExpand, setMaskExpand] = useState(10);
  const [maskBlur, setMaskBlur] = useState(18);
  const [workingLongSide, setWorkingLongSide] = useState(1024);
  const [cnEnabled, setCnEnabled] = useState(false);
  const [cnType, setCnType] = useState("canny");
  const [cnScale, setCnScale] = useState(0.45);
  const [enricherPreset, setEnricherPreset] = useState("general");
  const [protectFace, setProtectFace] = useState(false);
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [samMasks, setSamMasks] = useState<string[]>([]);
  const [samLabels, setSamLabels] = useState<string[]>([]);
  const [samSelected, setSamSelected] = useState(0);
  const [loading, setLoading] = useState(false);
  const [maskLoading, setMaskLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [enrichedPos, setEnrichedPos] = useState("");
  const [enrichedNeg, setEnrichedNeg] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (sentImage && !workingImage) { setWorkingImage(sentImage); onClearSent(); }
  }, [sentImage, workingImage, onClearSent]);

  // Load presets on mount
  useEffect(() => {
    getPresets().then(r => setPresets(r.presets)).catch(() => { });
  }, []);

  useEffect(() => {
    if (!prompt.trim()) { setEnrichedPos(""); setEnrichedNeg(""); return; }
    const t = setTimeout(async () => {
      try { const r = await enrichPreview(prompt, negative, autoEnrich, enricherPreset); setEnrichedPos(r.positive); setEnrichedNeg(r.negative); } catch { }
    }, 300);
    return () => clearTimeout(t);
  }, [prompt, negative, autoEnrich, enricherPreset]);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = () => { setWorkingImage((reader.result as string).split(",")[1]); setMaskImage(null); setResultImage(null); };
    reader.readAsDataURL(file);
  };

  const handleImageClick = async (e: React.MouseEvent<HTMLImageElement>) => {
    if (maskMode !== "click" || !workingImage || maskLoading) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    setMaskLoading(true); setStatus("🧠 SAM analyzing click...");
    try {
      const blob = await fetch(`data:image/png;base64,${workingImage}`).then(r => r.blob());
      const fd = new FormData();
      fd.append("image", blob, "image.png");
      fd.append("x", String(x));
      fd.append("y", String(y));
      const res = await clickMask(fd);
      if (res.status === "success" && res.masks && res.masks.length > 0) {
        setSamMasks(res.masks);
        setSamLabels(res.labels || ["S", "M", "L"]);
        const sel = res.selected ?? 0;
        setSamSelected(sel);
        setMaskImage(res.masks[sel]);
        setStatus(`✓ ${res.masks.length} masks found — use S/M/L to toggle size`);
      } else { setStatus(`✗ ${res.error || "No mask found"}`); }
    } catch (e: any) { setStatus(`SAM error: ${e.message}`); }
    setMaskLoading(false);
  };

  const handleAutoMask = async () => {
    if (!workingImage) return;
    setMaskLoading(true); setStatus("Generating mask...");
    try {
      const blob = await fetch(`data:image/png;base64,${workingImage}`).then(r => r.blob());
      const fd = new FormData();
      fd.append("image", blob, "image.png"); fd.append("target", maskTarget); fd.append("prompt", prompt);
      const res = await autoMask(fd);
      if (res.masks && res.masks.length > 0) { setMaskImage(res.masks[0]); setStatus(`✓ Mask created`); }
      else { setStatus(res.error || "No masks found"); }
    } catch (e: any) { setStatus(`Mask error: ${e.message}`); }
    setMaskLoading(false);
  };

  const handleCancel = async () => {
    try { await cancelInference(); } catch { }
  };

  const handleApply = async () => {
    if (!workingImage) { setStatus("Upload an image first"); return; }
    if (!maskImage) { setStatus("Create a mask first"); return; }
    if (!prompt.trim()) { setStatus("Enter a prompt"); return; }
    setLoading(true); setStatus(""); setResultImage(null);
    try {
      const imgBlob = await fetch(`data:image/png;base64,${workingImage}`).then(r => r.blob());
      const maskBlob = await fetch(`data:image/png;base64,${maskImage}`).then(r => r.blob());
      const fd = new FormData();
      fd.append("image", imgBlob, "image.png"); fd.append("mask", maskBlob, "mask.png");
      fd.append("prompt", prompt); fd.append("negative", negative);
      fd.append("auto_enrich", String(autoEnrich));
      fd.append("strength", String(strength)); fd.append("steps", String(editSteps));
      fd.append("guidance", String(guidance)); fd.append("seed", String(seed));
      fd.append("mask_expand", String(maskExpand)); fd.append("mask_blur", String(maskBlur));
      fd.append("working_long_side", String(workingLongSide));
      fd.append("use_controlnet", String(cnEnabled));
      fd.append("controlnet_type", cnType);
      fd.append("cn_scale", String(cnScale));
      fd.append("enricher_preset", enricherPreset);
      fd.append("protect_face", String(protectFace));
      const res = await editImage(fd);
      if (res.status === "success" && res.image) { setResultImage(res.image); setStatus(`✓ Done in ${res.elapsed}s · Seed: ${res.seed}`); }
      else if (res.status === "cancelled") { setStatus("⚠️ Cancelled"); }
      else { setStatus(`✗ ${res.error}`); }
    } catch (e: any) { setStatus(`Error: ${e.message}`); }
    setLoading(false);
  };

  return (
    <div className="fade-in">
      <h1 style={{ fontSize: "1.5rem", fontWeight: 700, marginBottom: 4 }}>✏️ Edit</h1>
      <p style={{ color: "var(--text-secondary)", marginBottom: 20, fontSize: "0.9rem" }}>Click to select · Auto mask · AI inpainting</p>

      {/* TOP ROW: Source+Overlay | Result */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
        {/* LEFT: Source image + mask overlay */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-secondary)" }}>Source Image</span>
            {workingImage && <button style={{ background: "none", border: "none", color: "var(--text-muted)", fontSize: "0.8rem", cursor: "pointer" }} onClick={() => { setWorkingImage(null); setMaskImage(null); setResultImage(null); }}>✕ Clear</button>}
          </div>
          <div className="image-display" style={{ minHeight: 400, position: "relative", cursor: workingImage ? (maskMode === "click" ? "crosshair" : "default") : "pointer" }}
            onClick={() => !workingImage && fileInputRef.current?.click()}>
            {workingImage ? (
              <>
                <img src={`data:image/png;base64,${workingImage}`} alt="Source" onClick={handleImageClick}
                  style={{ width: "100%", height: "100%", objectFit: "contain", cursor: maskMode === "click" ? "crosshair" : "default" }} />
                {maskImage && (
                  <img src={`data:image/png;base64,${maskImage}`} alt="Mask overlay"
                    style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "contain", opacity: 0.45, mixBlendMode: "screen", pointerEvents: "none" }} />
                )}
                {maskLoading && (
                  <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)", background: "rgba(0,0,0,0.7)", padding: "8px 16px", borderRadius: 8, color: "white", fontSize: "0.85rem" }}>
                    <span className="spinner" /> Analyzing...
                  </div>
                )}
              </>
            ) : (
              <label style={{ cursor: "pointer", color: "var(--text-muted)", textAlign: "center", padding: 40, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%" }}>
                <div style={{ fontSize: "3rem", marginBottom: 12, opacity: 0.3 }}>📤</div>
                <div>Click to upload or use Generate → Send to Editor</div>
              </label>
            )}
            <input ref={fileInputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={handleUpload} />
          </div>
          {maskMode === "click" && workingImage && (
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textAlign: "center" }}>
              👆 Click on the image to select a region with SAM AI
            </div>
          )}
        </div>

        {/* RIGHT: Result image */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-secondary)" }}>Result</span>
          <div className={`image-display ${loading ? "loading" : ""}`} style={{ minHeight: 400 }}>
            {resultImage ? (
              <img src={`data:image/jpeg;base64,${resultImage}`} alt="Result" className="scale-pop" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
            ) : (
              <div style={{ color: "var(--text-muted)", textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%" }}>
                <div style={{ fontSize: "3rem", marginBottom: 12, opacity: 0.3 }}>📸</div>
                <div>Edited result will appear here</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Progress + Status */}
      {loading && <ProgressBar task="edit" onCancel={handleCancel} />}
      {status && !loading && (
        <div className={`status-pill ${status.startsWith("✓") ? "success" : status.startsWith("✗") ? "error" : ""}`} style={{ marginBottom: 16 }}>{status}</div>
      )}

      {/* BOTTOM: Controls */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
        {/* Column 1: Prompt + Mask */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="glass-card" style={{ padding: 16 }}>
            <h3 style={{ fontSize: "0.85rem", fontWeight: 600, marginBottom: 10 }}>💬 Prompt</h3>
            <textarea className="input-field" rows={2} placeholder="Describe what to change..." value={prompt} onChange={(e) => setPrompt(e.target.value)} style={{ fontSize: "0.85rem" }} />
            <textarea className="input-field" rows={1} placeholder="Negative (optional)" value={negative} onChange={(e) => setNegative(e.target.value)} style={{ marginTop: 6, fontSize: "0.78rem" }} />
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
              <label style={{ display: "flex", alignItems: "center", gap: 5, fontSize: "0.8rem", color: "var(--text-secondary)", cursor: "pointer" }}>
                <input type="checkbox" checked={autoEnrich} onChange={(e) => setAutoEnrich(e.target.checked)} /> ✨ Auto-enrich
              </label>
            </div>
          </div>

          <div className="glass-card" style={{ padding: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <h3 style={{ fontSize: "0.85rem", fontWeight: 600 }}>🎭 Mask Mode</h3>
              <div style={{ display: "flex", gap: 4 }}>
                <button className={maskMode === "click" ? "btn-primary" : "btn-secondary"} style={{ padding: "3px 10px", fontSize: "0.72rem" }} onClick={() => setMaskMode("click")}>🖱️ Click (SAM)</button>
                <button className={maskMode === "auto" ? "btn-primary" : "btn-secondary"} style={{ padding: "3px 10px", fontSize: "0.72rem" }} onClick={() => setMaskMode("auto")}>🎯 Auto</button>
              </div>
            </div>
            {maskMode === "auto" && (
              <div style={{ display: "flex", gap: 6 }}>
                <select className="input-field" style={{ padding: "6px 10px", flex: 1, fontSize: "0.8rem" }} value={maskTarget} onChange={(e) => setMaskTarget(e.target.value)}>
                  <option value="top">Top (상의)</option>
                  <option value="pants">Pants/Skirt (하의)</option>
                  <option value="dress">Dress (원피스/전신)</option>
                  <option value="hair">Hair (머리)</option>
                  <option value="face">Face (얼굴)</option>
                  <option value="accessories">Accessories (모자/가방 등)</option>
                  <option value="background">Background (배경)</option>
                  <option value="person">Person (전체인물)</option>
                </select>
                <button className="btn-secondary" onClick={handleAutoMask} disabled={maskLoading || !workingImage} style={{ fontSize: "0.78rem" }}>
                  {maskLoading ? <span className="spinner" /> : "Generate"}
                </button>
              </div>
            )}
            {maskMode === "click" && (
              <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
                Click on the source image to select a region. SAM AI will detect the object boundary automatically.
                {samMasks.length > 0 && (
                  <div style={{ marginTop: 8, display: "flex", gap: 4, alignItems: "center" }}>
                    <span style={{ fontWeight: 600, marginRight: 4 }}>Size:</span>
                    {samMasks.map((_, i) => (
                      <button key={i}
                        className={samSelected === i ? "btn-primary" : "btn-secondary"}
                        style={{ padding: "3px 12px", fontSize: "0.72rem", fontWeight: 600, minWidth: 36 }}
                        onClick={() => { setSamSelected(i); setMaskImage(samMasks[i]); }}>
                        {samLabels[i] || ["S", "M", "L"][i]}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {maskImage && (
              <button className="btn-secondary" style={{ marginTop: 8, width: "100%", fontSize: "0.78rem", padding: "6px" }} onClick={() => { setMaskImage(null); setSamMasks([]); setSamLabels([]); setSamSelected(0); }}>🗑️ Clear Mask</button>
            )}
          </div>

          {/* Enricher Preset */}
          <div className="glass-card" style={{ padding: 16 }}>
            <h3 style={{ fontSize: "0.85rem", fontWeight: 600, marginBottom: 10 }}>🧪 Edit Preset</h3>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {presets.map(p => (
                <button key={p.id}
                  className={enricherPreset === p.id ? "btn-primary" : "btn-secondary"}
                  style={{ padding: "4px 10px", fontSize: "0.72rem", borderRadius: 16 }}
                  onClick={() => {
                    setEnricherPreset(p.id);
                    setStrength(p.strength);
                    setGuidance(p.guidance);
                    setEditSteps(p.steps);
                  }}>
                  {p.label}
                </button>
              ))}
            </div>
            {presets.find(p => p.id === enricherPreset) && (
              <div style={{ marginTop: 8, fontSize: "0.72rem", color: "var(--text-muted)", lineHeight: 1.4 }}>
                {presets.find(p => p.id === enricherPreset)!.description}
                <span style={{ marginLeft: 8, color: "var(--accent)", fontWeight: 500 }}>
                  S:{presets.find(p => p.id === enricherPreset)!.strength} / G:{presets.find(p => p.id === enricherPreset)!.guidance}
                </span>
              </div>
            )}
          </div>

          {enrichedPos && <EnrichedPreview label="📝 Final Positive" text={enrichedPos} />}
          {enrichedNeg && <EnrichedPreview label="🚫 Final Negative" text={enrichedNeg} />}
        </div>

        {/* Column 2: Settings */}
        <div className="glass-card" style={{ padding: 16 }}>
          <h3 style={{ fontSize: "0.85rem", fontWeight: 600, marginBottom: 10 }}>⚙️ Settings</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <Slider label="Strength" value={strength} min={0.3} max={0.95} step={0.05} onChange={setStrength} />
            <Slider label="Steps" value={editSteps} min={10} max={60} step={1} onChange={setEditSteps} />
            <Slider label="Guidance" value={guidance} min={1} max={12} step={0.5} onChange={setGuidance} />
            <Slider label="Resolution" value={workingLongSide} min={512} max={1536} step={64} onChange={setWorkingLongSide} />
            <Slider label="Mask Expand" value={maskExpand} min={0} max={30} step={1} onChange={setMaskExpand} />
            <Slider label="Mask Blur" value={maskBlur} min={0} max={40} step={1} onChange={setMaskBlur} />
            <div style={{ marginTop: 8 }}>
              <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.82rem", color: "var(--text-primary)", cursor: "pointer" }}>
                <input type="checkbox" checked={protectFace} onChange={(e) => setProtectFace(e.target.checked)} />
                🛡️ Protect Face & Hair
              </label>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginLeft: 24, marginTop: 2 }}>
                Prevents accidental changes to the face when editing clothing.
              </div>
            </div>
            <div style={{ marginTop: 8 }}><label style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: 3, display: "block" }}>Seed</label><input type="number" className="input-field" value={seed} onChange={(e) => setSeed(Number(e.target.value))} style={{ padding: "5px 8px", fontSize: "0.82rem" }} /></div>
          </div>
        </div>

        {/* Column 3: Apply */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="glass-card" style={{ padding: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: cnEnabled ? 10 : 0 }}>
              <label style={{ display: "flex", alignItems: "center", gap: 5, fontSize: "0.85rem", fontWeight: 600, cursor: "pointer" }}>
                <input type="checkbox" checked={cnEnabled} onChange={(e) => setCnEnabled(e.target.checked)} /> 🔗 ControlNet
              </label>
            </div>
            {cnEnabled && (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ display: "flex", gap: 4 }}>
                  {(["canny", "depth", "openpose"] as const).map(t => (
                    <button key={t} className={cnType === t ? "btn-primary" : "btn-secondary"}
                      style={{ padding: "4px 10px", fontSize: "0.72rem", flex: 1 }}
                      onClick={() => setCnType(t)}>
                      {t === "canny" ? "🔲 Canny" : t === "depth" ? "🌐 Depth" : "🦴 Pose"}
                    </button>
                  ))}
                </div>
                <div style={{ fontSize: "0.72rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
                  {cnType === "canny" && "🔲 Edge detection — preserves outlines and shapes"}
                  {cnType === "depth" && "🌐 Depth map — maintains spatial proportions"}
                  {cnType === "openpose" && "🦴 Skeleton — preserves human pose"}
                </div>
                <Slider label="Scale" value={cnScale} min={0.1} max={1} step={0.05} onChange={setCnScale} />
                <div style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>⚠️ ControlNet uses extra VRAM. First run downloads model.</div>
              </div>
            )}
          </div>

          <button className="btn-primary" onClick={handleApply} disabled={loading || !workingImage} style={{ padding: "14px 0", fontSize: "0.95rem" }}>
            {loading ? (<><span className="spinner" /> Applying...</>) : cnEnabled ? `▶️ Apply with ${cnType}` : "▶️ Apply Edit"}
          </button>
          {resultImage && (
            <button className="btn-secondary" style={{ width: "100%", fontSize: "0.85rem" }}
              onClick={() => { setWorkingImage(resultImage); setResultImage(null); setMaskImage(null); setStatus(""); }}>
              🔄 Use Result as Source
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <label style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>{label}</label>
        <span style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>{value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} style={{ width: "100%" }} />
    </div>
  );
}
