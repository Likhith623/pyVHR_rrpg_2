import { useState, useEffect, useRef } from "react";

// ── Palette ──────────────────────────────────────────────────
const BG     = "#04080f";
const CYAN   = "#00d4ff";
const PURPLE = "#a855f7";
const ORANGE = "#f97316";
const GREEN  = "#22c55e";
const GOLD   = "#eab308";
const RED    = "#ef4444";
const CARD   = "rgba(255,255,255,0.04)";
const BORDER = "rgba(255,255,255,0.09)";
const TEXT   = "#f1f5f9";
const DIM    = "#94a3b8";
const MUTED  = "#475569";

// ── Global CSS ────────────────────────────────────────────────
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800;900&family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:2px;}
.slide-in{animation:sIn .45s cubic-bezier(.16,1,.3,1) both;}
@keyframes sIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
@keyframes float-y{0%,100%{transform:translateY(0px)}50%{transform:translateY(-10px)}}
.ndot{transition:all .3s ease;cursor:pointer;}
.ndot:hover{transform:scale(1.5);}
.nav-btn{transition:all .2s;border:none;}
.nav-btn:hover:not(:disabled){opacity:.8;transform:scale(1.04);}
.nav-btn:disabled{cursor:not-allowed;}
`;

// ── Atoms ─────────────────────────────────────────────────────

function Card({ children, st, color }) {
  return (
    <div style={{
      background: CARD,
      border: `1px solid ${color ? color + "35" : BORDER}`,
      borderRadius: 14,
      padding: "16px 18px",
      boxShadow: color ? `0 4px 28px ${color}12` : "none",
      ...st
    }}>
      {children}
    </div>
  );
}

function Tag({ children, color = CYAN }) {
  return (
    <span style={{
      background: color + "18", color,
      border: `1px solid ${color}44`,
      borderRadius: 6, padding: "2px 9px",
      fontSize: 10, fontFamily: "'Space Mono',monospace",
      fontWeight: 700, letterSpacing: 1.2,
      textTransform: "uppercase", display: "inline-block"
    }}>
      {children}
    </span>
  );
}

function SLabel({ children, color = CYAN }) {
  return (
    <div style={{
      fontFamily: "'Space Mono',monospace", fontSize: 10,
      color, letterSpacing: 3, textTransform: "uppercase",
      marginBottom: 10, opacity: .85
    }}>
      {children}
    </div>
  );
}

function H1({ children, st = {} }) {
  return (
    <h1 style={{
      fontFamily: "'Syne',sans-serif", fontWeight: 900,
      fontSize: "clamp(22px,3.2vw,42px)", color: TEXT,
      lineHeight: 1.1, marginBottom: 10, ...st
    }}>
      {children}
    </h1>
  );
}

function H2({ children, color, st = {} }) {
  return (
    <h2 style={{
      fontFamily: "'Syne',sans-serif", fontWeight: 800,
      fontSize: "clamp(13px,1.6vw,17px)", color: color || TEXT,
      lineHeight: 1.2, marginBottom: 4, ...st
    }}>
      {children}
    </h2>
  );
}

function P({ children, st = {} }) {
  return (
    <p style={{
      fontFamily: "'Inter',sans-serif",
      fontSize: "clamp(11px,1.2vw,13.5px)",
      color: DIM, lineHeight: 1.75, ...st
    }}>
      {children}
    </p>
  );
}

function Mono({ children, color = MUTED, st = {} }) {
  return (
    <span style={{
      fontFamily: "'Space Mono',monospace",
      fontSize: 9, color, letterSpacing: .5, ...st
    }}>
      {children}
    </span>
  );
}

// ── Flowchart atoms ───────────────────────────────────────────

function FBox({ label, sub, color = CYAN, sm, w = "auto" }) {
  return (
    <div style={{
      background: color + "0f",
      border: `1.5px solid ${color}55`,
      borderRadius: 10,
      padding: sm ? "6px 10px" : "10px 14px",
      minWidth: sm ? 80 : 105,
      width: w,
      textAlign: "center",
      boxShadow: `0 2px 14px ${color}1a`,
      flexShrink: 0
    }}>
      <div style={{
        color, fontFamily: "'Space Mono',monospace",
        fontWeight: 700, fontSize: sm ? 9 : 11, lineHeight: 1.4
      }}>
        {label}
      </div>
      {sub && (
        <div style={{
          color: MUTED, fontSize: 8.5, marginTop: 2,
          fontFamily: "'Inter',sans-serif", lineHeight: 1.3
        }}>
          {sub}
        </div>
      )}
    </div>
  );
}

function Arr({ color = CYAN, v = false, label = "" }) {
  return (
    <div style={{
      color, opacity: .6, flexShrink: 0,
      display: "flex", flexDirection: "column",
      alignItems: "center", gap: 1,
      padding: v ? "2px 0" : "0 2px"
    }}>
      {label && (
        <span style={{
          fontSize: 7.5, fontFamily: "'Space Mono',monospace",
          opacity: .9, letterSpacing: 1
        }}>
          {label}
        </span>
      )}
      <span style={{ fontSize: v ? 14 : 16 }}>{v ? "↓" : "→"}</span>
    </div>
  );
}

// Vertical single-track flow
function VFlow({ nodes, color = CYAN }) {
  return (
    <div style={{
      display: "flex", flexDirection: "column",
      alignItems: "center", gap: 0
    }}>
      {nodes.map((n, i) => (
        <span key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <FBox label={n.label} sub={n.sub} color={n.color || color} sm={n.sm !== false} />
          {i < nodes.length - 1 && <Arr color={n.color || color} v />}
        </span>
      ))}
    </div>
  );
}

// ── Slide wrapper ─────────────────────────────────────────────

function SW({ children, st = {} }) {
  return (
    <div
      className="slide-in"
      style={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        padding: "clamp(18px,3vh,40px) clamp(20px,5vw,62px)",
        overflowY: "auto",
        backgroundImage: "radial-gradient(circle,rgba(255,255,255,0.035) 1px,transparent 1px)",
        backgroundSize: "30px 30px",
        ...st
      }}
    >
      {children}
    </div>
  );
}

// ── SLIDE 0 — TITLE ──────────────────────────────────────────

function Slide0() {
  return (
    <SW st={{ justifyContent: "center", position: "relative" }}>
      {/* Decorative circles */}
      {[340, 250, 150].map((r, i) => (
        <div key={i} style={{
          position: "absolute", right: "6%", top: "50%",
          width: r, height: r, borderRadius: "50%",
          transform: "translateY(-50%)", pointerEvents: "none",
          border: `1px solid ${[CYAN, PURPLE, CYAN][i]}${["14", "22", "40"][i]}`,
          display: "flex", alignItems: "center", justifyContent: "center"
        }}>
          {i === 2 && (
            <div style={{ fontSize: 54, animation: "float-y 4s ease-in-out infinite" }}>
              🧠
            </div>
          )}
        </div>
      ))}

      <div style={{ maxWidth: 560, position: "relative", zIndex: 1 }}>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 22 }}>
          {[
            { t: "Deepfake Detection", c: GREEN },
            { t: "4-Model Ensemble", c: PURPLE },
          ].map((x, i) => <Tag key={i} color={x.c}>{x.t}</Tag>)}
        </div>

        <div style={{
          fontFamily: "'Syne',sans-serif", fontWeight: 900,
          fontSize: "clamp(58px,10vw,108px)", lineHeight: .92,
          background: `linear-gradient(140deg,${CYAN} 0%,${PURPLE} 55%,${ORANGE} 100%)`,
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
        }}>
          Neuro<br />Pulse
        </div>

        <P st={{
          fontSize: "clamp(12px,1.3vw,15px)", marginTop: 20,
          lineHeight: 1.9, maxWidth: 440
        }}>
          A multi-model AI system that detects{" "}
          <span style={{ color: CYAN }}>deepfake videos</span>{" "}
          by simultaneously analysing heartbeat signals, visual patterns, frequency artifacts,
          and transformer attention — then fusing all four expert verdicts into one final answer.
        </P>

        <div style={{ display: "flex", gap: 9, flexWrap: "wrap", marginTop: 20 }}>
          {[
            { t: "4 AI Models", c: GREEN },
            { t: "117 Physiological Features", c: CYAN },
            { t: "Ensemble Fusion", c: PURPLE },
            { t: "FF++ · CelebDF · DFDC", c: ORANGE },
          ].map((x, i) => <Tag key={i} color={x.c}>{x.t}</Tag>)}
        </div>

        <div style={{
          marginTop: 26, paddingTop: 18, borderTop: `1px solid ${BORDER}`,
          fontFamily: "'Space Mono',monospace", fontSize: 9,
          color: MUTED, letterSpacing: 1.5
        }}>
          Department of Computer Science &amp; Engineering · Use ← → keys or dots to navigate
        </div>
      </div>
    </SW>
  );
}

// ── SLIDE 1 — THE PROBLEM ────────────────────────────────────

function Slide1() {
  return (
    <SW>
      <SLabel color={ORANGE}>01 · The Problem</SLabel>
      <H1>Deepfakes Are Getting<br /><span style={{ color: ORANGE }}>Dangerously Convincing</span></H1>
      <P st={{ maxWidth: 620, marginBottom: 20 }}>
        A deepfake is a video where AI has replaced or synthesised someone's face — so realistic
        that humans cannot tell it apart from real footage. The technology is growing fast and being
        misused everywhere.
      </P>

      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(195px,1fr))",
        gap: 12, marginBottom: 18
      }}>
        {[
          { icon: "🎭", title: "What IS a deepfake?", body: "AI generates a fake face on top of real video footage. The person never said or did those things — but it looks perfectly real.", color: ORANGE },
          { icon: "📰", title: "Fake news at scale", body: "Politicians, scientists, celebrities — anyone can be made to say anything. Millions can be misled in hours.", color: PURPLE },
          { icon: "💸", title: "Financial fraud", body: "Real companies have lost millions because employees were tricked by deepfake video calls pretending to be the CEO.", color: CYAN },
          { icon: "📈", title: "Exploding growth", body: "Deepfake videos have grown 900%+ in 2 years. Detection tools are struggling far behind the creators.", color: GREEN },
        ].map((c, i) => (
          <Card key={i} color={c.color} st={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ fontSize: 26 }}>{c.icon}</div>
            <H2 color={c.color}>{c.title}</H2>
            <P st={{ fontSize: 12, margin: 0 }}>{c.body}</P>
          </Card>
        ))}
      </div>

      <Card color={ORANGE} st={{ background: ORANGE + "08", maxWidth: 700 }}>
        <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
          <div style={{ fontSize: 26, flexShrink: 0 }}>🤔</div>
          <div>
            <H2 color={ORANGE} st={{ marginBottom: 6 }}>Why do existing detectors fail?</H2>
            <P st={{ fontSize: 12.5, margin: 0 }}>
              Most detection tools watch only <strong style={{ color: TEXT }}>one signal</strong> —
              pixel noise, motion blur, or colour shifts. Deepfake creators simply improve that
              one thing and immediately break the detector. We need to look at the video from{" "}
              <strong style={{ color: TEXT }}>many different angles at once</strong> to reliably
              catch fakes.
            </P>
          </div>
        </div>
      </Card>
    </SW>
  );
}

// ── SLIDE 2 — OUR SOLUTION ───────────────────────────────────

function Slide2() {
  return (
    <SW>
      <SLabel>02 · Our Solution</SLabel>
      <H1>NeuroPulse: <span style={{ color: CYAN }}>4 Specialists,</span><br />One Final Verdict</H1>
      <P st={{ maxWidth: 640, marginBottom: 20 }}>
        Instead of one detector that can be fooled, NeuroPulse runs 4 completely different AI
        models in parallel. Each expert looks for different clues. Then they all vote together —
        like a jury.
      </P>

      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(190px,1fr))",
        gap: 12, marginBottom: 18
      }}>
        {[
          { icon: "💓", num: "M1", name: "rPPG", color: GREEN, desc: "Measures heartbeat from tiny face colour changes. Deepfakes have no real blood flow — this signal is absent or incoherent." },
          { icon: "🔭", num: "M2", name: "EfficientNet-B4", color: CYAN, desc: "Looks at 16 video frames. Tracks how visual anomalies evolve over time using BiLSTM + Attention." },
          { icon: "🌊", num: "M3", name: "Xception", color: PURPLE, desc: "Detects JPEG compression artifacts and frequency patterns left behind by all deepfake generators." },
          { icon: "🏗️", num: "M4", name: "Swin Transformer", color: ORANGE, desc: "Hierarchical window attention + DCT frequency branch. Most powerful visual architecture in the suite." },
        ].map((m, i) => (
          <Card key={i} color={m.color} st={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <div style={{ fontSize: 26 }}>{m.icon}</div>
              <div>
                <Mono color={m.color} st={{ letterSpacing: 1 }}>{m.num}</Mono>
                <H2 color={m.color} st={{ fontSize: 14, margin: 0 }}>{m.name}</H2>
              </div>
            </div>
            <P st={{ fontSize: 12, margin: 0 }}>{m.desc}</P>
          </Card>
        ))}
      </div>

      {/* Mini fusion diagram */}
      <Card st={{ background: GOLD + "07", border: `1px solid ${GOLD}24` }}>
        <div style={{
          display: "flex", alignItems: "center", gap: 5,
          flexWrap: "wrap", justifyContent: "center"
        }}>
          {[
            { l: "rPPG", c: GREEN }, { l: "CNN", c: CYAN },
            { l: "Xception", c: PURPLE }, { l: "Swin", c: ORANGE }
          ].map((s, i) => (
            <span key={i} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <FBox label={s.l + " Score"} color={s.c} sm w="100px" />
              <Arr color={GOLD} />
            </span>
          ))}
          <FBox label="Weighted Ensemble" sub="Soft-vote fusion" color={GOLD} w="148px" />
          <Arr color={GOLD} />
          <div style={{
            background: GOLD + "1a", border: `2px solid ${GOLD}77`,
            borderRadius: 12, padding: "10px 20px",
            fontFamily: "'Syne',sans-serif", fontWeight: 800,
            color: GOLD, fontSize: 16, textAlign: "center",
            boxShadow: `0 0 22px ${GOLD}22`
          }}>
            REAL<br />or FAKE?
          </div>
        </div>
      </Card>
    </SW>
  );
}

// ── SLIDE 3 — DATASET ────────────────────────────────────────

function Slide3() {
  return (
    <SW>
      <SLabel color={PURPLE}>03 · Dataset</SLabel>
      <H1>Training on the <span style={{ color: PURPLE }}>Hardest Benchmarks</span> Available</H1>
      <P st={{ maxWidth: 620, marginBottom: 18 }}>
        We combined 3 standard research datasets, each representing a different generation
        of deepfake technology — from early methods to high-realism modern fakes.
      </P>

      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(220px,1fr))",
        gap: 13, marginBottom: 16
      }}>
        {[
          {
            icon: "🎬", name: "FaceForensics++", year: "ICCV 2019 · Rossler et al.",
            desc: "1,000 real YouTube videos each faked using 4 methods: DeepFakes, Face2Face, FaceSwap, NeuralTextures.",
            pills: ["4 manipulation methods", "Most used benchmark", "Multiple quality levels"],
            color: CYAN
          },
          {
            icon: "⭐", name: "Celeb-DF v2", year: "CVPR 2020 · Li et al.",
            desc: "590 celebrity deepfakes made with a much more realistic improved synthesis pipeline.",
            pills: ["High visual realism", "Celebrity subjects", "Harder than FF++"],
            color: PURPLE
          },
          {
            icon: "🏆", name: "DFDC Sample", year: "Facebook AI · Dolhansky et al.",
            desc: "Sample from Facebook's Deepfake Detection Challenge — diverse actors, lighting, environments.",
            pills: ["Real-world diversity", "Various compression", "Professional quality"],
            color: ORANGE
          },
        ].map((d, i) => (
          <Card key={i} color={d.color}>
            <div style={{ fontSize: 26, marginBottom: 7 }}>{d.icon}</div>
            <H2 color={d.color} st={{ marginBottom: 2 }}>{d.name}</H2>
            <Mono st={{ display: "block", marginBottom: 8 }}>{d.year}</Mono>
            <P st={{ fontSize: 12, margin: 0, marginBottom: 9 }}>{d.desc}</P>
            {d.pills.map((p, j) => (
              <div key={j} style={{
                fontFamily: "'Space Mono',monospace", fontSize: 8.5,
                color: d.color, opacity: .8, marginTop: 3
              }}>▸ {p}</div>
            ))}
          </Card>
        ))}
      </div>

      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(158px,1fr))",
        gap: 10
      }}>
        {[
          { icon: "✂️", label: "Train/Test Split", value: "80% train · 20% test" },
          { icon: "🔒", label: "Identity-Aware", value: "Same person never in both sets" },
          { icon: "⚖️", label: "Class Balanced", value: "Equal real & fake per source" },
          { icon: "💾", label: "GPU-Safe Budget", value: "Max 200 videos / source (P100)" },
          { icon: "🎲", label: "Reproducible", value: "Random seed = 42 everywhere" },
        ].map((s, i) => (
          <Card key={i} st={{ textAlign: "center", display: "flex", flexDirection: "column", gap: 5 }}>
            <div style={{ fontSize: 20 }}>{s.icon}</div>
            <Mono st={{ display: "block", letterSpacing: 1 }}>{s.label}</Mono>
            <div style={{ fontFamily: "'Inter',sans-serif", fontSize: 11.5, color: DIM }}>{s.value}</div>
          </Card>
        ))}
      </div>
    </SW>
  );
}

// ── SLIDE 4 — MODEL 1: rPPG ──────────────────────────────────

function Slide4() {
  return (
    <SW>
      <SLabel color={GREEN}>04 · Model 1 — Physiological Signal Detection</SLabel>
      <H1>rPPG: Detecting the <span style={{ color: GREEN }}>Heartbeat</span> in the Video</H1>
      <P st={{ maxWidth: 600, marginBottom: 16 }}>
        When your heart beats, blood flows to your face causing tiny colour changes invisible to
        humans but measurable by AI. <strong style={{ color: TEXT }}>Deepfakes have no real
        heartbeat</strong> — so this signal is flat, noisy, or spatially incoherent.
      </P>

      <div style={{ display: "grid", gridTemplateColumns: "3fr 2fr", gap: 16, flex: 1 }}>
        {/* Left info */}
        <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10
          }}>
            {[
              { icon: "🗺️", title: "9 Face Zones", desc: "Forehead, left/right cheeks, chin, nose, jaw — each region tracked separately for spatial coherence analysis" },
              { icon: "📊", title: "117 Features", desc: "Heart rate variability (RMSSD, SDNN, pNN50), spectral SNR, cross-ROI correlation, geometry — 117 dimensions" },
              { icon: "📈", title: "CHROM Method", desc: "Chrominance-based rPPG algorithm (de Haan & Jeanne 2013). Bandpass filtered 0.7–4.0 Hz (42–240 BPM)" },
              { icon: "🤖", title: "Stacking Ensemble", desc: "XGBoost + LightGBM + HistGradientBoosting → Logistic Regression meta-learner (cv=5, passthrough=False)" },
            ].map((f, i) => (
              <Card key={i} st={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <div style={{ fontSize: 20 }}>{f.icon}</div>
                <H2 color={GREEN} st={{ fontSize: 12 }}>{f.title}</H2>
                <P st={{ fontSize: 11, margin: 0 }}>{f.desc}</P>
              </Card>
            ))}
          </div>

          <Card color={GREEN} st={{ background: GREEN + "07" }}>
            <H2 color={GREEN} st={{ fontSize: 12, marginBottom: 7 }}>🔬 Why deepfakes fail this test</H2>
            {[
              "Real face: ALL 9 zones pulse at the same cardiac frequency — spatially coherent",
              "Real face: HRV metrics (RMSSD, SDNN, pNN50) follow genuine nervous system patterns",
              "Deepfake: Zones show different frequencies — no synchronisation between forehead & cheeks",
              "Deepfake: No real cardiac pulse → signal is flat, noise-only, or artificially cyclic",
            ].map((b, i) => (
              <div key={i} style={{ display: "flex", gap: 7, alignItems: "flex-start", marginTop: 5 }}>
                <span style={{ color: GREEN, flexShrink: 0 }}>▸</span>
                <P st={{ fontSize: 11, margin: 0 }}>{b}</P>
              </div>
            ))}
          </Card>
        </div>

        {/* Flowchart */}
        <Card st={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <Mono st={{ letterSpacing: 2, marginBottom: 12, display: "block", textAlign: "center" }}>
            PIPELINE FLOWCHART
          </Mono>
          <VFlow color={GREEN} nodes={[
            { label: "🎬 Video Input", sub: "MP4 · max 60 frames (linspace)" },
            { label: "MediaPipe FaceMesh", sub: "468 landmarks · temporal tracking" },
            { label: "9 Face ROI Zones", sub: "Forehead · Cheeks · Chin · Nose · Jaw" },
            { label: "CHROM rPPG Signal", sub: "Bandpass 0.7–4.0 Hz · detrended" },
            { label: "117 Feature Vector", sub: "HRV + Spectral + Phase + Geometry" },
            { label: "RobustScaler + ExtraTrees", sub: "Feature selection · percentile clip" },
            { label: "Stacking Ensemble v7.2", sub: "XGB · LGBM · HistBoost → LogReg" },
            { label: "✅  P_rPPG Score", sub: "0 = Real · 1 = Fake probability" },
          ]} />
        </Card>
      </div>
    </SW>
  );
}

// ── SLIDE 5 — MODEL 2: EfficientNet ─────────────────────────

function Slide5() {
  return (
    <SW>
      <SLabel color={CYAN}>05 · Model 2 — Spatio-Temporal Deep Learning</SLabel>
      <H1>EfficientNet-B4 + <span style={{ color: CYAN }}>BiLSTM + Attention</span></H1>
      <P st={{ maxWidth: 600, marginBottom: 16 }}>
        This model looks at <strong style={{ color: TEXT }}>16 video frames</strong> — what each
        frame looks like (spatial) AND how things change between frames (temporal). Deepfakes often
        show inconsistent facial textures or unnatural frame transitions.
      </P>

      <div style={{ display: "grid", gridTemplateColumns: "3fr 2fr", gap: 16, flex: 1 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {[
              { icon: "🔍", title: "EfficientNet-B4", desc: "Compound-scaled CNN. 1792-dim spatial feature per frame. Pretrained ImageNet. drop_path=0.2" },
              { icon: "🔁", title: "2-Layer BiLSTM", desc: "Reads frames forward AND backward. 512-dim temporal encoding. cuDNN disabled for P100" },
              { icon: "👁️", title: "4-Head Attention", desc: "Learns which frames are most suspicious. Key-padding mask ignores padded frames" },
              { icon: "🔬", title: "MTCNN + Cache", desc: "Face detected at 224×224, saved as .npy files on disk — keeps GPU RAM free" },
            ].map((f, i) => (
              <Card key={i} st={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <div style={{ fontSize: 20 }}>{f.icon}</div>
                <H2 color={CYAN} st={{ fontSize: 12 }}>{f.title}</H2>
                <P st={{ fontSize: 11, margin: 0 }}>{f.desc}</P>
              </Card>
            ))}
          </div>

          <Card color={CYAN} st={{ background: CYAN + "07" }}>
            <H2 color={CYAN} st={{ fontSize: 12, marginBottom: 7 }}>🔧 Training tricks</H2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 5 }}>
              {[
                ["Focal Loss α=0.6", "Focuses training on hard examples"],
                ["MixUp λ~Beta(1,1)", "Blends two training videos together"],
                ["SWA from epoch 30", "Weight averaging for stable generalisation"],
                ["5-Pass TTA", "Original + flip + bright± + blur"],
                ["Grad accumulation ×4", "Effective batch = 8 (physical = 2)"],
                ["Cosine warmup LR", "10% warmup then cosine decay"],
              ].map(([k, v], i) => (
                <div key={i} style={{ display: "flex", gap: 6, alignItems: "flex-start" }}>
                  <span style={{ color: CYAN, flexShrink: 0, fontSize: 10 }}>▸</span>
                  <div>
                    <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, color: CYAN }}>{k}</div>
                    <div style={{ fontFamily: "'Inter',sans-serif", fontSize: 10, color: DIM }}>{v}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Flowchart */}
        <Card st={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <Mono st={{ letterSpacing: 2, marginBottom: 12, display: "block", textAlign: "center" }}>
            PIPELINE FLOWCHART
          </Mono>
          <VFlow color={CYAN} nodes={[
            { label: "🎬 Video", sub: "From master dataset index CSV" },
            { label: "MTCNN Face Detect", sub: "224×224 · 16 frames sampled" },
            { label: "Face Cache (.npy)", sub: "Disk-saved · P100 RAM safe" },
            { label: "EfficientNet-B4", sub: "Spatial: 1792-dim per frame" },
            { label: "Unfreeze epoch 5", sub: "Backbone LR = full LR / 10" },
            { label: "2-Layer BiLSTM", sub: "Bidirectional → 512-dim output" },
            { label: "4-Head MH Attention", sub: "Padding mask applied" },
            { label: "Classifier Head", sub: "LayerNorm → GELU → Dropout 0.5" },
            { label: "✅  P_CNN Score", sub: "5-pass TTA · SWA averaged" },
          ]} />
        </Card>
      </div>
    </SW>
  );
}

// ── SLIDE 6 — MODEL 3: XCEPTION ─────────────────────────────

function Slide6() {
  return (
    <SW>
      <SLabel color={PURPLE}>06 · Model 3 — Dual-Branch Architecture</SLabel>
      <H1>Xception + <span style={{ color: PURPLE }}>Frequency Branch</span> + Hard Mining</H1>
      <P st={{ maxWidth: 600, marginBottom: 16 }}>
        Deepfake generators leave <strong style={{ color: TEXT }}>hidden signatures in the
        frequency domain</strong> — JPEG compression artifacts, spectral anomalies. This model has
        TWO parallel streams: one for appearance, one for frequency. They fuse at the end.
      </P>

      <div style={{ display: "grid", gridTemplateColumns: "3fr 2fr", gap: 16, flex: 1 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {[
              { icon: "🔬", title: "Xception 2048-dim", desc: "Depthwise separable convolutions. 2048-dim spatial features. Native 299×299 Xception resolution" },
              { icon: "👁️", title: "ECA Channel Attention", desc: "Efficient Channel Attention re-weights all 2048 channels — highlights the most informative ones" },
              { icon: "🌊", title: "Frequency Branch", desc: "Parallel 256-dim stream processes spatial features for JPEG compression artifact signatures" },
              { icon: "⛏️", title: "Hard Negative Mining", desc: "From epoch 10: finds examples near the decision boundary and trains twice as hard on those" },
            ].map((f, i) => (
              <Card key={i} st={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <div style={{ fontSize: 20 }}>{f.icon}</div>
                <H2 color={PURPLE} st={{ fontSize: 12 }}>{f.title}</H2>
                <P st={{ fontSize: 11, margin: 0 }}>{f.desc}</P>
              </Card>
            ))}
          </div>

          <Card color={PURPLE} st={{ background: PURPLE + "07" }}>
            <H2 color={PURPLE} st={{ fontSize: 12, marginBottom: 7 }}>🎭 Special training techniques</H2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 5 }}>
              {[
                ["MixUp α=0.2", "Blends two video sequences smoothly"],
                ["CutMix α=1.0", "Pastes patches between two videos"],
                ["Dynamic Focal α", "Computed from real/fake ratio per fold"],
                ["6-Pass TTA", "Adds 93% centre-crop for scale robustness"],
                ["Eye-Aligned Face", "Laplacian sharpness filter applied"],
                ["9 Adam param groups", "Fine-grained LR control per layer"],
              ].map(([k, v], i) => (
                <div key={i} style={{ display: "flex", gap: 6, alignItems: "flex-start" }}>
                  <span style={{ color: PURPLE, flexShrink: 0, fontSize: 10 }}>▸</span>
                  <div>
                    <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, color: PURPLE }}>{k}</div>
                    <div style={{ fontFamily: "'Inter',sans-serif", fontSize: 10, color: DIM }}>{v}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Dual-branch flowchart */}
        <Card st={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 4 }}>
          <Mono st={{ letterSpacing: 2, marginBottom: 8, display: "block", textAlign: "center" }}>
            DUAL-BRANCH FLOWCHART
          </Mono>

          <FBox label="🎬 Video — Eye-Aligned" sub="299×299 face · MTCNN" color={PURPLE} sm />
          <Arr color={PURPLE} v />
          <FBox label="Xception Backbone" sub="2048-dim features/frame" color={PURPLE} />
          <Arr color={PURPLE} v />
          <FBox label="ECA Channel Attention" sub="Re-weights 2048 channels" color={PURPLE} />
          <Arr color={PURPLE} v />

          {/* Branch */}
          <div style={{ display: "flex", gap: 8, alignItems: "flex-start", justifyContent: "center" }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
              <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 8, color: PURPLE, opacity: .75, letterSpacing: 1 }}>TEMPORAL</div>
              <FBox label="BiLSTM × 2" sub="512-dim" color={PURPLE} sm />
              <Arr color={PURPLE} v />
              <FBox label="MHA 4 heads" sub="Avg-pool" color={PURPLE} sm />
            </div>
            <div style={{ color: MUTED, fontSize: 18, alignSelf: "center", opacity: .35 }}>⊕</div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
              <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 8, color: PURPLE, opacity: .75, letterSpacing: 1 }}>FREQUENCY</div>
              <FBox label="Freq Branch" sub="Parallel 256-dim" color={PURPLE} sm />
              <Arr color={PURPLE} v />
              <FBox label="Artifact Patterns" sub="JPEG signatures" color={PURPLE} sm />
            </div>
          </div>

          <Arr color={PURPLE} v label="CONCAT" />
          <FBox label="Fused: 768-dim" sub="512 temporal + 256 freq" color={PURPLE} />
          <Arr color={PURPLE} v />
          <FBox label="✅  P_CNN Score" sub="6-pass TTA · threshold 0.5" color={PURPLE} sm />
        </Card>
      </div>
    </SW>
  );
}

// ── SLIDE 7 — MODEL 4: SWIN ──────────────────────────────────

function Slide7() {
  return (
    <SW>
      <SLabel color={ORANGE}>07 · Model 4 — Vision Transformer</SLabel>
      <H1>Swin Transformer + <span style={{ color: ORANGE }}>On-the-fly DCT Branch</span></H1>
      <P st={{ maxWidth: 600, marginBottom: 16 }}>
        Swin Transformer splits each image into patches and applies
        <strong style={{ color: TEXT }}> self-attention within shifted local windows</strong>.
        It captures subtle structural anomalies CNNs miss. Plus a real-time DCT frequency branch
        for compression artifact detection.
      </P>

      <div style={{ display: "grid", gridTemplateColumns: "3fr 2fr", gap: 16, flex: 1 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {[
              { icon: "🪟", title: "Shifted Windows", desc: "Hierarchical attention in 7×7 patch windows, then shifted to share info across boundaries. 768-dim output" },
              { icon: "📐", title: "On-the-fly DCT", desc: "Converts each frame to 64×64 DCT coefficients at runtime. 128 block statistics → 192-dim encoder output" },
              { icon: "📚", title: "Full 5-Fold OOF", desc: "All 5 cross-validation folds trained in one session. The most rigorous validation in our pipeline" },
              { icon: "🎓", title: "Progressive Training", desc: "Curriculum: 5 frames in epochs 0-4, then 10, then full 16. Easier to start, harder as training continues" },
            ].map((f, i) => (
              <Card key={i} st={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <div style={{ fontSize: 20 }}>{f.icon}</div>
                <H2 color={ORANGE} st={{ fontSize: 12 }}>{f.title}</H2>
                <P st={{ fontSize: 11, margin: 0 }}>{f.desc}</P>
              </Card>
            ))}
          </div>

          <Card color={ORANGE} st={{ background: ORANGE + "07" }}>
            <H2 color={ORANGE} st={{ fontSize: 12, marginBottom: 7 }}>⚙️ Engineering details</H2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 5 }}>
              {[
                ["pack_padded_sequence", "Clean gradient flow in BiLSTM"],
                ["DCT as register_buffer", "Pre-computed matrix · zero overhead"],
                ["Progressive curriculum", "5 → 10 → 16 frames over epochs"],
                ["Focal α=0.5 (bug fix)", "Was 0.75 — biased toward fakes"],
                ["Orthogonal LSTM init", "Forget gate bias forced to 1.0"],
                ["Fold-skip detection", "Resumes from saved .pth checkpoints"],
              ].map(([k, v], i) => (
                <div key={i} style={{ display: "flex", gap: 6, alignItems: "flex-start" }}>
                  <span style={{ color: ORANGE, flexShrink: 0, fontSize: 10 }}>▸</span>
                  <div>
                    <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, color: ORANGE }}>{k}</div>
                    <div style={{ fontFamily: "'Inter',sans-serif", fontSize: 10, color: DIM }}>{v}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Swin flowchart */}
        <Card st={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 4 }}>
          <Mono st={{ letterSpacing: 2, marginBottom: 8, display: "block", textAlign: "center" }}>
            DUAL-BRANCH FLOWCHART
          </Mono>

          <FBox label="🎬 Video" sub="224×224 · eye-aligned" color={ORANGE} sm />
          <Arr color={ORANGE} v />
          <FBox label="Swin-Tiny Backbone" sub="768-dim · drop_path=0.2" color={ORANGE} />
          <Arr color={ORANGE} v />
          <FBox label="ECA Channel Attention" sub="Re-weights 768 channels" color={ORANGE} />
          <Arr color={ORANGE} v />

          {/* Branch */}
          <div style={{ display: "flex", gap: 8, alignItems: "flex-start", justifyContent: "center" }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
              <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 8, color: ORANGE, opacity: .75, letterSpacing: 1 }}>VISUAL</div>
              <FBox label="pack_padded BiLSTM" sub="512-dim · clean grads" color={ORANGE} sm />
              <Arr color={ORANGE} v />
              <FBox label="MHA 4 heads" sub="Masked avg-pool" color={ORANGE} sm />
            </div>
            <div style={{ color: MUTED, fontSize: 18, alignSelf: "center", opacity: .35 }}>⊕</div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
              <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 8, color: ORANGE, opacity: .75, letterSpacing: 1 }}>DCT</div>
              <FBox label="On-the-fly DCT" sub="64×64 → 128 stats" color={ORANGE} sm />
              <Arr color={ORANGE} v />
              <FBox label="Freq Encoder" sub="192-dim output" color={ORANGE} sm />
            </div>
          </div>

          <Arr color={ORANGE} v label="CONCAT" />
          <FBox label="Fused: 704-dim" sub="512 temporal + 192 DCT" color={ORANGE} />
          <Arr color={ORANGE} v />
          <FBox label="✅  P_CNN Score" sub="5-fold OOF · 6-pass TTA" color={ORANGE} sm />
        </Card>
      </div>
    </SW>
  );
}

// ── SLIDE 8 — ENSEMBLE FUSION ────────────────────────────────

function Slide8() {
  return (
    <SW>
      <SLabel color={GOLD}>08 · Ensemble Fusion</SLabel>
      <H1>All 4 Experts Vote — <span style={{ color: GOLD }}>The Jury Decides</span></H1>
      <P st={{ maxWidth: 660, marginBottom: 18 }}>
        Each model outputs a probability (0 = real, 1 = fake). These 4 scores are combined with
        a weighted average — models that scored higher on validation data get more say. This
        ensemble consistently beats any single model alone.
      </P>

      <Card st={{ background: GOLD + "06", border: `1px solid ${GOLD}20`, marginBottom: 15 }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>

          {/* 4 model cards */}
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center" }}>
            {[
              { name: "Model 1: rPPG", type: "Physiological", score: "P_rPPG", color: GREEN, icon: "💓" },
              { name: "Model 2: EfficientNet", type: "Spatio-Temporal CNN", score: "P_CNN", color: CYAN, icon: "🔭" },
              { name: "Model 3: Xception", type: "Freq + Visual CNN", score: "P_CNN", color: PURPLE, icon: "🌊" },
              { name: "Model 4: Swin", type: "Transformer + DCT", score: "P_CNN", color: ORANGE, icon: "🏗️" },
            ].map((m, i) => (
              <div key={i} style={{
                background: m.color + "0b", border: `1px solid ${m.color}30`,
                borderRadius: 12, padding: "12px 16px", minWidth: 115,
                display: "flex", flexDirection: "column", alignItems: "center", gap: 6
              }}>
                <div style={{ fontSize: 22 }}>{m.icon}</div>
                <H2 color={m.color} st={{ fontSize: 11, textAlign: "center", margin: 0 }}>{m.name}</H2>
                <Mono st={{ textAlign: "center" }}>{m.type}</Mono>
                <div style={{
                  background: m.color + "22", border: `1px solid ${m.color}44`,
                  borderRadius: 5, padding: "3px 12px",
                  fontFamily: "'Space Mono',monospace", fontSize: 10,
                  color: m.color, fontWeight: 700
                }}>{m.score}</div>
              </div>
            ))}
          </div>

          {/* Arrow */}
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
            <div style={{ color: GOLD, fontSize: 20, opacity: .6 }}>↓</div>
            <Mono color={GOLD} st={{ letterSpacing: 2 }}>WEIGHTED AVERAGE FUSION</Mono>
          </div>

          {/* Formula */}
          <div style={{
            background: GOLD + "0e", border: `1.5px solid ${GOLD}40`,
            borderRadius: 10, padding: "10px 28px",
            fontFamily: "'Space Mono',monospace", fontSize: 11,
            color: GOLD, textAlign: "center"
          }}>
            final = w₁·P_rPPG + w₂·P_EfficientNet + w₃·P_Xception + w₄·P_Swin
          </div>

          <div style={{ color: GOLD, fontSize: 20, opacity: .6 }}>↓</div>

          {/* Decision */}
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center" }}>
            <div style={{
              background: GREEN + "18", border: `2px solid ${GREEN}66`,
              borderRadius: 12, padding: "12px 28px",
              fontFamily: "'Syne',sans-serif", fontWeight: 800,
              fontSize: 16, color: GREEN, textAlign: "center",
              boxShadow: `0 0 22px ${GREEN}18`
            }}>
              ✅ REAL VIDEO
              <div style={{ fontSize: 9, fontFamily: "'Space Mono',monospace", color: MUTED, fontWeight: 400, marginTop: 3 }}>
                final score &lt; 0.5
              </div>
            </div>
            <div style={{
              background: ORANGE + "18", border: `2px solid ${ORANGE}66`,
              borderRadius: 12, padding: "12px 28px",
              fontFamily: "'Syne',sans-serif", fontWeight: 800,
              fontSize: 16, color: ORANGE, textAlign: "center",
              boxShadow: `0 0 22px ${ORANGE}18`
            }}>
              ⚠️ DEEPFAKE
              <div style={{ fontSize: 9, fontFamily: "'Space Mono',monospace", color: MUTED, fontWeight: 400, marginTop: 3 }}>
                final score ≥ 0.5
              </div>
            </div>
          </div>
        </div>
      </Card>

      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(185px,1fr))", gap: 11
      }}>
        {[
          { icon: "🧩", title: "Why it works", desc: "Each model catches a different type of fake. Together they cover all the ways deepfakes can be detected.", color: GOLD },
          { icon: "⚖️", title: "Weighted by merit", desc: "Models with higher validation AUC during training earn a larger weight in the final fusion.", color: CYAN },
          { icon: "🛡️", title: "Failure protection", desc: "If one model is tricked, the other three can outvote it and still produce the right answer.", color: GREEN },
        ].map((c, i) => (
          <Card key={i} color={c.color} st={{ display: "flex", gap: 10 }}>
            <div style={{ fontSize: 20, flexShrink: 0 }}>{c.icon}</div>
            <div>
              <H2 color={c.color} st={{ fontSize: 12, marginBottom: 4 }}>{c.title}</H2>
              <P st={{ fontSize: 12, margin: 0 }}>{c.desc}</P>
            </div>
          </Card>
        ))}
      </div>
    </SW>
  );
}

// ── SLIDE 9 — RESULTS + FUTURE ───────────────────────────────

function Slide9() {
  return (
    <SW>
      <SLabel>09 · Results, Limitations & Future Work</SLabel>
      <H1>What We Built, What's <span style={{ color: PURPLE }}>Still Hard</span>,<br />and Where We Go Next</H1>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, flex: 1, marginTop: 14 }}>
        <Card color={GREEN} st={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <H2 color={GREEN}>✅ What we built</H2>
          {[
            "Full 4-model deepfake detection pipeline on 3 public datasets",
            "rPPG model: 117 physiological features from 9 face zones",
            "3 deep learning models with BiLSTM + Multi-Head Attention",
            "Frequency analysis in 2 models: Xception (256d) + Swin (DCT 192d)",
            "Weighted ensemble fusion for the final robust prediction",
            "5-fold cross-validation — unbiased, rigorous evaluation",
            "Grad-CAM visualisations — shows what each model is looking at",
            "Identity-aware splits — no data leakage between train and test",
          ].map((a, i) => (
            <div key={i} style={{ display: "flex", gap: 7, alignItems: "flex-start" }}>
              <span style={{ color: GREEN, flexShrink: 0 }}>▸</span>
              <P st={{ fontSize: 12, margin: 0 }}>{a}</P>
            </div>
          ))}
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Card color={ORANGE} st={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <H2 color={ORANGE}>⚠️ Limitations</H2>
            {[
              "Dataset limited by Kaggle P100 GPU — only ~200 videos per source",
              "rPPG struggles with compressed or low frame-rate videos",
              "Models may not catch entirely NEW types of deepfakes not in training data",
              "No real-time inference yet — runs offline only",
              "No voice/audio analysis — lip-sync fakes not covered",
            ].map((l, i) => (
              <div key={i} style={{ display: "flex", gap: 7, alignItems: "flex-start" }}>
                <span style={{ color: ORANGE, flexShrink: 0 }}>▸</span>
                <P st={{ fontSize: 12, margin: 0 }}>{l}</P>
              </div>
            ))}
          </Card>

          <Card color={CYAN} st={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <H2 color={CYAN}>🚀 Future work</H2>
            {[
              "Add audio-visual fusion for voice deepfakes + lip-sync analysis",
              "Real-time browser-based detection tool for everyday users",
              "Train on larger datasets using cloud TPUs",
              "Adapt to diffusion-model and next-generation deepfakes",
            ].map((f, i) => (
              <div key={i} style={{ display: "flex", gap: 7, alignItems: "flex-start" }}>
                <span style={{ color: CYAN, flexShrink: 0 }}>▸</span>
                <P st={{ fontSize: 12, margin: 0 }}>{f}</P>
              </div>
            ))}
          </Card>
        </div>
      </div>

      <div style={{
        marginTop: 18, textAlign: "center",
        fontFamily: "'Syne',sans-serif", fontWeight: 900,
        fontSize: "clamp(14px,2.3vw,22px)",
        background: `linear-gradient(135deg,${CYAN},${PURPLE},${ORANGE})`,
        WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
      }}>
        "NeuroPulse: Because your eyes can be fooled — AI does not have to be."
      </div>
    </SW>
  );
}

// ── Slides registry ───────────────────────────────────────────

const SLIDES = [Slide0, Slide1, Slide2, Slide3, Slide4, Slide5, Slide6, Slide7, Slide8, Slide9];
const LABELS = [
  "NeuroPulse", "The Problem", "Our Solution", "Dataset",
  "Model 1 · rPPG", "Model 2 · EfficientNet", "Model 3 · Xception",
  "Model 4 · Swin", "Ensemble Fusion", "Results & Future"
];
const DOT_COLORS = [CYAN, ORANGE, CYAN, PURPLE, GREEN, CYAN, PURPLE, ORANGE, GOLD, PURPLE];

// ── Navigation dots ───────────────────────────────────────────

function NavDots({ slide, go }) {
  return (
    <div style={{
      position: "fixed", right: 16, top: "50%", transform: "translateY(-50%)",
      display: "flex", flexDirection: "column", gap: 9, zIndex: 300
    }}>
      {LABELS.map((label, i) => (
        <div
          key={i}
          className="ndot"
          title={label}
          onClick={() => go(i)}
          style={{
            width: i === slide ? 9 : 6,
            height: i === slide ? 9 : 6,
            borderRadius: "50%",
            background: i === slide ? DOT_COLORS[i] : BORDER,
            boxShadow: i === slide ? `0 0 10px ${DOT_COLORS[i]}aa` : "none"
          }}
        />
      ))}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────

export default function NeuroPulse() {
  const [slide, setSlide] = useState(0);
  const ref = useRef(0);
  const total = SLIDES.length;

  const go = (n) => {
    if (n < 0 || n >= total) return;
    ref.current = n;
    setSlide(n);
  };

  // Keyboard navigation
  useEffect(() => {
    const h = (e) => {
      if (["ArrowRight", "ArrowDown", " "].includes(e.key)) {
        e.preventDefault();
        go(Math.min(ref.current + 1, total - 1));
      }
      if (["ArrowLeft", "ArrowUp"].includes(e.key)) {
        e.preventDefault();
        go(Math.max(ref.current - 1, 0));
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  const Comp = SLIDES[slide];
  const accent = DOT_COLORS[slide];

  return (
    <div style={{
      width: "100vw", height: "100vh", background: BG,
      position: "fixed", top: 0, left: 0, overflow: "hidden",
      fontFamily: "'Inter',sans-serif"
    }}>
      <style>{CSS}</style>

      {/* Progress bar */}
      <div style={{
        position: "fixed", top: 0, left: 0, right: 0,
        height: 2, background: BORDER, zIndex: 400
      }}>
        <div style={{
          width: `${((slide + 1) / total) * 100}%`, height: "100%",
          background: `linear-gradient(90deg,${accent},${PURPLE})`,
          transition: "width .4s ease",
          boxShadow: `0 0 8px ${accent}88`
        }} />
      </div>

      {/* Top bar */}
      <div style={{
        position: "fixed", top: 0, left: 0, right: 0, height: 34,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 40px", zIndex: 300
      }}>
        <Mono st={{ letterSpacing: 2, textTransform: "uppercase", fontSize: 9 }}>
          NEUROPULSE · {LABELS[slide]}
        </Mono>
        <Mono st={{ letterSpacing: 2, fontSize: 9 }}>
          {String(slide + 1).padStart(2, "0")} / {String(total).padStart(2, "0")}
        </Mono>
      </div>

      {/* Slide */}
      <div style={{ width: "100%", height: "100%", paddingTop: 34 }}>
        <Comp key={slide} />
      </div>

      {/* Side dots */}
      <NavDots slide={slide} go={go} />

      {/* Bottom nav */}
      <div style={{
        position: "fixed", bottom: 16, left: "50%",
        transform: "translateX(-50%)",
        display: "flex", gap: 10, zIndex: 300
      }}>
        {[
          { label: "← PREV", fn: () => go(slide - 1), dis: slide === 0 },
          { label: "NEXT →", fn: () => go(slide + 1), dis: slide === total - 1 },
        ].map((b, i) => (
          <button
            key={i}
            className="nav-btn"
            onClick={b.fn}
            disabled={b.dis}
            style={{
              background: b.dis ? CARD : accent + "22",
              border: `1px solid ${b.dis ? BORDER : accent + "66"}`,
              borderRadius: 8,
              color: b.dis ? MUTED : accent,
              fontFamily: "'Space Mono',monospace",
              fontSize: 10, padding: "6px 18px",
              opacity: b.dis ? .35 : 1,
              letterSpacing: 1, cursor: b.dis ? "not-allowed" : "pointer"
            }}
          >
            {b.label}
          </button>
        ))}
      </div>

      {/* Keyboard hint */}
      <div style={{
        position: "fixed", bottom: 16, right: 38,
        fontFamily: "'Space Mono',monospace",
        fontSize: 8, color: MUTED, letterSpacing: 2, zIndex: 300
      }}>
        ← → KEYS
      </div>
    </div>
  );
}