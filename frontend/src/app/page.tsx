"use client";

import {
  forwardRef,
  PointerEvent as ReactPointerEvent,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import {
  cancelJob,
  editImage,
  generateImage,
  getConfig,
  getStatus,
  ImageResult,
  JobStatus,
  ModelProfile,
  ProfileId,
  StudioConfig,
  StudioStatus,
  unloadModel,
} from "@/lib/api";

type Mode = "generate" | "edit";
type MaskHandle = { clear: () => void; exportMask: () => Promise<Blob | null>; hasMask: () => boolean };

const MaskCanvas = forwardRef<MaskHandle, { src: string; enabled: boolean; brush: number }>(
  function MaskCanvas({ src, enabled, brush }, ref) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const drawing = useRef(false);
    const painted = useRef(false);
    const lastPoint = useRef<{ x: number; y: number } | null>(null);
    const [ratio, setRatio] = useState(1);

    useEffect(() => {
      const image = new window.Image();
      image.onload = () => {
        const scale = Math.min(1, 1200 / Math.max(image.naturalWidth, image.naturalHeight));
        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
        canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
        setRatio(image.naturalWidth / image.naturalHeight);
        canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
        painted.current = false;
      };
      image.src = src;
    }, [src]);

    useImperativeHandle(ref, () => ({
      clear() {
        const canvas = canvasRef.current;
        if (canvas) canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
        painted.current = false;
      },
      exportMask() {
        const canvas = canvasRef.current;
        if (!canvas || !painted.current) return Promise.resolve(null);
        return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      },
      hasMask: () => painted.current,
    }));

    const point = (event: ReactPointerEvent<HTMLCanvasElement>) => {
      const canvas = event.currentTarget;
      const bounds = canvas.getBoundingClientRect();
      return {
        x: (event.clientX - bounds.left) * (canvas.width / bounds.width),
        y: (event.clientY - bounds.top) * (canvas.height / bounds.height),
      };
    };

    const draw = (event: ReactPointerEvent<HTMLCanvasElement>) => {
      if (!drawing.current || !lastPoint.current) return;
      const canvas = event.currentTarget;
      const next = point(event);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const bounds = canvas.getBoundingClientRect();
      ctx.strokeStyle = "#ffffff";
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = brush * (canvas.width / bounds.width);
      ctx.beginPath();
      ctx.moveTo(lastPoint.current.x, lastPoint.current.y);
      ctx.lineTo(next.x, next.y);
      ctx.stroke();
      lastPoint.current = next;
      painted.current = true;
    };

    return (
      <div
        className={`mask-stage ${enabled ? "is-painting" : ""}`}
        style={{
          aspectRatio: ratio,
          width: ratio >= 1 ? "100%" : "auto",
          height: ratio < 1 ? "100%" : "auto",
        }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt="Edit source" />
        <canvas
          ref={canvasRef}
          aria-label="Paint the area allowed to change"
          onPointerDown={(event) => {
            if (!enabled) return;
            drawing.current = true;
            const start = point(event);
            lastPoint.current = start;
            const canvas = event.currentTarget;
            const bounds = canvas.getBoundingClientRect();
            const ctx = canvas.getContext("2d");
            if (ctx) {
              ctx.fillStyle = "#ffffff";
              ctx.beginPath();
              ctx.arc(start.x, start.y, (brush * canvas.width / bounds.width) / 2, 0, Math.PI * 2);
              ctx.fill();
              painted.current = true;
            }
            event.currentTarget.setPointerCapture(event.pointerId);
          }}
          onPointerMove={draw}
          onPointerUp={() => { drawing.current = false; lastPoint.current = null; }}
          onPointerCancel={() => { drawing.current = false; lastPoint.current = null; }}
        />
      </div>
    );
  },
);

function Slider({ label, value, min, max, step = 1, onChange }: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (value: number) => void;
}) {
  return (
    <label className="slider-row">
      <span>{label}<strong>{value}</strong></span>
      <input type="range" value={value} min={min} max={max} step={step} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

const PROFILE_IDS: ProfileId[] = ["quality", "balanced", "fast"];

function ProfileSelector({ profiles, selected, onChange, disabled = false }: {
  profiles?: Record<ProfileId, ModelProfile>;
  selected: ProfileId;
  onChange: (profile: ProfileId) => void;
  disabled?: boolean;
}) {
  return (
    <div className="profile-picker" aria-label="Performance profile">
      {PROFILE_IDS.map((id) => {
        const profile = profiles?.[id];
        return (
          <button
            key={id}
            className={selected === id ? "selected" : ""}
            aria-pressed={selected === id}
            disabled={disabled}
            onClick={() => onChange(id)}
          >
            <span>{profile?.label || id}</span>
            <b>{profile?.steps || "–"} steps</b>
            {profile?.gated && <i>HF token</i>}
          </button>
        );
      })}
      {profiles && (
        <p>
          {profiles[selected].description} First download ≈ {profiles[selected].download_gb} GB · {profiles[selected].license}
          {profiles[selected].cached && " · cached"}
        </p>
      )}
    </div>
  );
}

function ResultView({ result, empty, busy }: { result: ImageResult | null; empty: string; busy: boolean }) {
  if (result) {
    const src = `data:image/png;base64,${result.image}`;
    return (
      <div className="result-wrap">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="result-image" src={src} alt="AI result" />
        <div className="result-meta">
          <span>{result.width} × {result.height}</span>
          <span>infer {result.timings.inference}s</span>
          {result.timings.load > 0 && <span>load {result.timings.load}s</span>}
          <span>seed {result.seed}</span>
          <a href={src} download={`morrow-${result.seed}.png`}>Download PNG</a>
        </div>
      </div>
    );
  }
  return (
    <div className={`empty-result ${busy ? "is-busy" : ""}`}>
      <div className="orb" />
      <p>{busy ? "The local model is working" : empty}</p>
    </div>
  );
}

const PROGRESS_PHASES = [
  { label: "Model", stages: ["downloading", "loading"] },
  { label: "Prepare", stages: ["preparing"] },
  { label: "Denoise", stages: ["inference"] },
  { label: "Finish", stages: ["composite", "encoding", "done"] },
];

function formatDuration(seconds: number | null | undefined) {
  if (seconds == null) return "Estimating…";
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function ProgressDock({ job, onCancel }: { job: JobStatus | null; onCancel: () => void }) {
  const stage = job?.active ? job.stage : "loading";
  const activeIndex = Math.max(0, PROGRESS_PHASES.findIndex((phase) => phase.stages.includes(stage)));
  const indeterminate = !job?.active || job.indeterminate;
  const percent = job?.active ? job.overall_progress : 2;
  const message = job?.active ? job.message : "Starting local job…";

  return (
    <div className="job-dock" role="status" aria-live="polite">
      <div className="job-head">
        <span className="job-spinner" />
        <div>
          <small>{job?.task === "edit" ? "IMAGE EDIT" : "IMAGE GENERATION"}</small>
          <b>{message}</b>
        </div>
        <strong className="job-percent">{indeterminate ? "•••" : `${Math.round(percent)}%`}</strong>
        <button disabled={!job?.cancellable} onClick={onCancel}>
          {job?.cancellable ? "Cancel" : "Loading…"}
        </button>
      </div>

      <div className="phase-line">
        {PROGRESS_PHASES.map((phase, index) => (
          <div key={phase.label} className={`${index < activeIndex ? "complete" : ""} ${index === activeIndex ? "active" : ""}`}>
            <span>{index < activeIndex ? "✓" : index + 1}</span>
            <b>{phase.label}</b>
          </div>
        ))}
      </div>

      <div className={`progress-track ${indeterminate ? "indeterminate" : ""}`}>
        <span style={{ width: indeterminate ? "32%" : `${percent}%` }} />
      </div>

      <div className="job-stats">
        <span>{job?.step && job.total ? `Step ${job.step} / ${job.total}` : stage === "downloading" ? "First-use model download" : "Preparing components"}</span>
        <span>{formatDuration(job?.elapsed)} elapsed</span>
        <span>{stage === "inference" ? `${formatDuration(job?.eta_seconds)} remaining` : indeterminate ? "Duration varies by cache/network" : "Almost there"}</span>
      </div>
    </div>
  );
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("generate");
  const [status, setStatus] = useState<StudioStatus | null>(null);
  const [config, setConfig] = useState<StudioConfig | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [generateProfile, setGenerateProfile] = useState<ProfileId>("fast");
  const [generatePrompt, setGeneratePrompt] = useState("");
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [generateSeed, setGenerateSeed] = useState(-1);
  const [generateResult, setGenerateResult] = useState<ImageResult | null>(null);

  const [editProfile, setEditProfile] = useState<ProfileId>("fast");
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [sourceUrl, setSourceUrl] = useState("");
  const [editPrompt, setEditPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [editSeed, setEditSeed] = useState(-1);
  const [useMask, setUseMask] = useState(false);
  const [brush, setBrush] = useState(48);
  const [feather, setFeather] = useState(8);
  const [editResult, setEditResult] = useState<ImageResult | null>(null);
  const maskRef = useRef<MaskHandle>(null);

  useEffect(() => {
    getConfig().then((value) => {
      setConfig(value);
      setGenerateProfile(value.default_profile);
      setEditProfile(value.default_profile);
    }).catch(() => undefined);
    let stopped = false;
    let timer = 0;
    const poll = async () => {
      let delay = 3000;
      try {
        const next = await getStatus();
        if (!stopped) setStatus(next);
        delay = next.job.active ? 500 : 3000;
      } catch {
        if (!stopped) setStatus(null);
      }
      if (!stopped) timer = window.setTimeout(poll, delay);
    };
    void poll();
    return () => { stopped = true; window.clearTimeout(timer); };
  }, []);

  useEffect(() => () => { if (sourceUrl) URL.revokeObjectURL(sourceUrl); }, [sourceUrl]);

  const run = async (action: () => Promise<ImageResult>, onResult: (result: ImageResult) => void) => {
    setBusy(true);
    setError("");
    try { onResult(await action()); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Unknown error"); }
    finally { setBusy(false); }
  };

  const submitGenerate = () => run(
    () => generateImage({ prompt: generatePrompt, width, height, profile: generateProfile, seed: generateSeed }),
    setGenerateResult,
  );

  const submitEdit = async () => {
    if (!sourceFile) return;
    const form = new FormData();
    form.append("image", sourceFile);
    form.append("prompt", editPrompt);
    form.append("negative", negative);
    form.append("profile", editProfile);
    form.append("seed", String(editSeed));
    form.append("feather", String(feather));
    if (useMask) {
      const mask = await maskRef.current?.exportMask();
      if (mask) form.append("mask", mask, "mask.png");
    }
    await run(() => editImage(form), setEditResult);
  };

  const chooseSource = (file?: File) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Choose a supported image file");
      return;
    }
    const maxUploadMb = config?.max_upload_mb || 32;
    if (file.size > maxUploadMb * 1024 * 1024) {
      setError(`Image exceeds the ${maxUploadMb} MB upload limit`);
      return;
    }
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    setError("");
    setSourceFile(file);
    setSourceUrl(URL.createObjectURL(file));
    setEditResult(null);
    maskRef.current?.clear();
  };

  const activeJob = status?.job.active || busy;
  const selectedId = mode === "generate" ? generateProfile : editProfile;
  const selectedProfile = config?.profiles[mode][selectedId];
  const selectedModel = selectedProfile?.transformer_id || selectedProfile?.model_id || "Loading model catalog…";
  const accessBlocked = Boolean(
    selectedProfile?.gated && !selectedProfile.cached && !config?.hf_token_configured,
  );

  return (
    <main>
      <header className="topbar">
        <div className="brand"><span className="brand-mark">M</span><div><b>Morrow</b><small>local image studio</small></div></div>
        <nav className="mode-switch">
          <button aria-pressed={mode === "generate"} disabled={activeJob} className={mode === "generate" ? "active" : ""} onClick={() => setMode("generate")}>Generate</button>
          <button aria-pressed={mode === "edit"} disabled={activeJob} className={mode === "edit" ? "active" : ""} onClick={() => setMode("edit")}>Edit</button>
        </nav>
        <div className="runtime">
          <span className={`status-dot ${status ? "online" : ""}`} />
          <div><b>{status?.hardware.gpu_name || "Backend offline"}</b><small>{status ? `${status.hardware.vram_allocated_gb} / ${status.hardware.vram_total_gb} GB VRAM` : "Start the local API"}</small></div>
          <button disabled={activeJob} title="Unload active model" onClick={() => unloadModel().catch((e) => setError(e.message))}>Unload</button>
        </div>
      </header>

      <section className="model-strip">
        <span>MODEL</span>
        <b>{selectedModel}</b>
        <i>{selectedProfile?.steps || "–"} steps · {selectedProfile?.long_side || "–"}px</i>
        <i>{status?.model.loaded ? status.model.attention_backend : config?.attention_backend || "native"} · 4-bit transformer + text encoder</i>
        <i className="checker">No runtime safety checker</i>
      </section>

      {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")}>Dismiss</button></div>}
      {accessBlocked && (
        <div className="access-warning" role="alert">
          This profile needs accepted Hugging Face model terms and an HF_TOKEN in .env.local before its first download.
        </div>
      )}

      {mode === "generate" ? (
        <section className="workspace">
          <aside className="control-panel">
            <div className="section-heading"><span>01</span><div><h2>Create</h2><p>Describe the image you want.</p></div></div>
            <label className="field-label">Performance</label>
            <ProfileSelector profiles={config?.profiles.generate} selected={generateProfile} onChange={setGenerateProfile} disabled={activeJob} />
            <label className="field-label" htmlFor="generate-prompt">Prompt</label>
            <textarea id="generate-prompt" maxLength={4000} value={generatePrompt} onChange={(e) => setGeneratePrompt(e.target.value)} placeholder="A cinematic portrait in soft window light…" rows={8} />
            <div className="preset-grid">
              {[[1024, 1024, "Square"], [1152, 768, "Landscape"], [768, 1152, "Portrait"]].map(([w, h, label]) => (
                <button key={String(label)} aria-pressed={width === w && height === h} disabled={activeJob} className={width === w && height === h ? "selected" : ""} onClick={() => { setWidth(Number(w)); setHeight(Number(h)); }}>
                  <span style={{ aspectRatio: `${w}/${h}` }} />{label}
                </button>
              ))}
            </div>
            <label className="field-label compact" htmlFor="generate-seed">Seed</label>
            <input id="generate-seed" min={-1} max={2 ** 31 - 1} className="number-input" type="number" value={generateSeed} onChange={(e) => setGenerateSeed(Number(e.target.value))} />
            <button className="primary-action" disabled={activeJob || accessBlocked || !generatePrompt.trim()} onClick={submitGenerate}>{activeJob ? "Working…" : "Generate image"}</button>
          </aside>
          <div className="canvas-panel"><ResultView result={generateResult} busy={activeJob} empty="Your image will appear here" /></div>
        </section>
      ) : (
        <section className="workspace edit-workspace">
          <aside className="control-panel">
            <div className="section-heading"><span>01</span><div><h2>Instruction</h2><p>Say only what should change.</p></div></div>
            <label className="field-label">Performance</label>
            <ProfileSelector profiles={config?.profiles.edit} selected={editProfile} onChange={setEditProfile} disabled={activeJob} />
            <label className="field-label" htmlFor="edit-prompt">Edit instruction</label>
            <textarea id="edit-prompt" maxLength={4000} value={editPrompt} onChange={(e) => setEditPrompt(e.target.value)} placeholder="Change the jacket to dark green leather. Keep the person, pose and background unchanged." rows={6} />
            <details>
              <summary>Advanced settings</summary>
              <label className="field-label compact" htmlFor="negative-prompt">Negative prompt</label>
              <input id="negative-prompt" maxLength={4000} value={negative} onChange={(e) => setNegative(e.target.value)} placeholder="Optional" />
              <label className="field-label compact" htmlFor="edit-seed">Seed</label>
              <input id="edit-seed" min={-1} max={2 ** 31 - 1} className="number-input" type="number" value={editSeed} onChange={(e) => setEditSeed(Number(e.target.value))} />
            </details>
            <button className="primary-action" disabled={activeJob || !sourceFile || !editPrompt.trim()} onClick={submitEdit}>{activeJob ? "Working…" : "Apply edit"}</button>
          </aside>

          <div className="editor-panel">
            <div className="editor-toolbar">
              <label className="upload-button">{sourceFile ? "Replace image" : "Choose image"}<input disabled={activeJob} type="file" accept="image/*" onChange={(event) => chooseSource(event.target.files?.[0])} /></label>
              <button aria-pressed={useMask} className={useMask ? "active" : ""} disabled={!sourceFile || activeJob} onClick={() => setUseMask((value) => !value)}>Brush mask</button>
              {useMask && <><label>Brush <input type="range" min="12" max="160" value={brush} onChange={(e) => setBrush(Number(e.target.value))} /></label><button onClick={() => maskRef.current?.clear()}>Clear</button></>}
            </div>
            <div className="source-stage">
              {sourceUrl ? <MaskCanvas ref={maskRef} src={sourceUrl} enabled={useMask} brush={brush} /> : <label className="drop-empty">Choose an image to begin<input disabled={activeJob} type="file" accept="image/*" onChange={(event) => chooseSource(event.target.files?.[0])} /></label>}
            </div>
            {useMask && <div className="mask-options"><span>Only painted pixels may change. Everything else is copied from the original.</span><Slider label="Edge feather" value={feather} min={0} max={32} onChange={setFeather} /></div>}
          </div>

          <div className="canvas-panel edit-result"><ResultView result={editResult} busy={activeJob} empty="Edited result" /></div>
        </section>
      )}

      {activeJob && (
        <ProgressDock
          job={status?.job.active ? status.job : null}
          onCancel={() => cancelJob().catch(() => undefined)}
        />
      )}
    </main>
  );
}
