/**
 * frontend/src/components/ScoreDisplay.jsx
 *
 * Cinematic results reveal:
 *  - Animated counter for the main probability score
 *  - Staggered branch reveal (one card at a time, 200ms apart)
 *  - Dynamic counter per branch (counts 0 → final value)
 *  - Neon glow on verdict: green = REAL, red = FAKE
 */

"use client";

import { useEffect, useState, useRef } from "react";
import styles from "./ScoreDisplay.module.css";

// ── Meta ──────────────────────────────────────────────────────────────────────
const VERDICT_META = {
  REAL:      { color: "var(--neon-green)", glow: "var(--neon-green-glow)", emoji: "✓", label: "AUTHENTIC" },
  FAKE:      { color: "var(--neon-red)",   glow: "var(--neon-red-glow)",   emoji: "✗", label: "DEEPFAKE"  },
  UNCERTAIN: { color: "var(--neon-amber)", glow: "var(--neon-amber-glow)", emoji: "?", label: "UNCERTAIN" },
};

const BRANCH_META = {
  Temporal_Spatial_Flow:       { icon: "🎞️", label: "Temporal-Spatial Flow",        sub: "3D-CNN · R3D-18"         },
  Audio_Visual_Sync:           { icon: "🔊", label: "Audio-Visual Sync",             sub: "Dual-Encoder SyncNet"    },
  Spatial_Frequency_Artifacts: { icon: "📡", label: "Spatial-Frequency Artifacts",   sub: "FFT + ResNet-18"         },
  Biological_rPPG:             { icon: "💓", label: "Biological rPPG",               sub: "LSTM Pulse Analysis"     },
};

// ── Animated counter hook ──────────────────────────────────────────────────────
function useCountUp(target, duration = 1200, delay = 0) {
  const [value, setValue] = useState(0);
  const raf = useRef(null);

  useEffect(() => {
    const timeout = setTimeout(() => {
      const start     = performance.now();
      const startVal  = 0;

      const tick = (now) => {
        const elapsed  = now - start;
        const progress = Math.min(elapsed / duration, 1);
        // Ease-out cubic
        const eased    = 1 - Math.pow(1 - progress, 3);
        setValue(Math.round(startVal + eased * (target - startVal)));
        if (progress < 1) raf.current = requestAnimationFrame(tick);
      };

      raf.current = requestAnimationFrame(tick);
    }, delay);

    return () => {
      clearTimeout(timeout);
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [target, duration, delay]);

  return value;
}

// ── Branch Card (self-contained stagger + counter) ────────────────────────────
function BranchCard({ branchKey, score, index }) {
  const [visible, setVisible]   = useState(false);
  const STAGGER_DELAY            = 350; // ms between cards
  const APPEAR_DELAY             = index * STAGGER_DELAY;
  const COUNT_DELAY              = APPEAR_DELAY + 300; // start counting slightly after card appears

  const pct      = Math.round(score * 100);
  const animated = useCountUp(pct, 900, COUNT_DELAY);

  const isFake   = score >= 0.65;
  const isMid    = score >= 0.35 && score < 0.65;
  const barColor = isFake ? "var(--neon-red)" : isMid ? "var(--neon-amber)" : "var(--neon-green)";
  const barGlow  = isFake ? "var(--neon-red-glow)" : isMid ? "var(--neon-amber-glow)" : "var(--neon-green-glow)";

  const meta = BRANCH_META[branchKey] ?? { icon: "🔬", label: branchKey, sub: "" };

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), APPEAR_DELAY);
    return () => clearTimeout(t);
  }, [APPEAR_DELAY]);

  return (
    <div
      className={`${styles.branchCard} ${visible ? styles.branchVisible : ""}`}
      style={{ "--bar-color": barColor, "--bar-glow": barGlow, "--delay": `${APPEAR_DELAY}ms` }}
    >
      {/* Top row */}
      <div className={styles.branchTop}>
        <span className={styles.branchIcon}>{meta.icon}</span>
        <div>
          <p className={styles.branchLabel}>{meta.label}</p>
          <p className={styles.branchSub}>{meta.sub}</p>
        </div>
        <span className={styles.branchPct} style={{ color: barColor }}>
          {animated}%
        </span>
      </div>

      {/* Bar */}
      <div className={styles.barTrack}>
        <div
          className={styles.barFill}
          style={{
            width: visible ? `${pct}%` : "0%",
            background: barColor,
            boxShadow: visible ? `0 0 8px ${barGlow}` : "none",
            transitionDelay: `${COUNT_DELAY}ms`,
          }}
        />
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────
export default function ScoreDisplay({ result }) {
  if (!result) return null;

  const { deepfake_probability, verdict, branch_scores, processing_time_ms, job_id } = result;

  const meta    = VERDICT_META[verdict] ?? VERDICT_META.UNCERTAIN;
  const pct     = Math.round(deepfake_probability * 100);
  const animated = useCountUp(pct, 1400, 200);

  return (
    <section className={styles.card} aria-live="polite">

      {/* ── Verdict Header ──────────────────────────────────────────────── */}
      <div className={styles.verdictHeader} style={{ "--verdict-glow": meta.glow }}>
        <div className={styles.verdictBadge} style={{ color: meta.color, borderColor: meta.color }}>
          <span className={styles.verdictSymbol}>{meta.emoji}</span>
        </div>
        <div>
          <h2 className={styles.verdictLabel} style={{ color: meta.color }}>
            {meta.label}
          </h2>
          <p className={styles.jobId}>
            <span className={styles.jobIdDot} style={{ background: meta.color }} />
            JOB {job_id?.slice(0, 8).toUpperCase() ?? "N/A"}
          </p>
        </div>
      </div>

      {/* ── Main Probability Gauge ─────────────────────────────────────── */}
      <div className={styles.gaugeSection}>
        {/* Circular gauge */}
        <div
          className={styles.circleOuter}
          style={{ "--pct": pct, "--clr": meta.color, "--glow": meta.glow }}
          aria-hidden="true"
        >
          <svg className={styles.circleSvg} viewBox="0 0 120 120">
            {/* Track */}
            <circle cx="60" cy="60" r="52" className={styles.circleTrack} />
            {/* Fill — stroke-dasharray animated via CSS */}
            <circle
              cx="60" cy="60" r="52"
              className={styles.circleFill}
              style={{
                stroke: meta.color,
                filter: `drop-shadow(0 0 6px ${meta.glow})`,
                strokeDasharray: `${(pct / 100) * 327} 327`,
              }}
            />
          </svg>
          <div className={styles.circleInner}>
            <span className={styles.circleValue} style={{ color: meta.color }}>{animated}</span>
            <span className={styles.circlePct}>%</span>
          </div>
        </div>

        {/* Linear gauge */}
        <div className={styles.linearSection}>
          <p className={styles.linearLabel}>Deepfake Probability</p>
          <div className={styles.linearTrack}>
            <div
              className={styles.linearFill}
              style={{
                width: `${pct}%`,
                background: `linear-gradient(90deg, ${meta.color}aa, ${meta.color})`,
                boxShadow: `0 0 12px ${meta.glow}`,
              }}
            />
          </div>
          <div className={styles.linearTicks}>
            {[0, 25, 50, 75, 100].map(t => (
              <span key={t} className={styles.tick}>{t}%</span>
            ))}
          </div>
          <p className={styles.processingTime}>
            Analysed in {processing_time_ms?.toLocaleString() ?? "–"} ms
          </p>
        </div>
      </div>

      {/* ── Branch Breakdown ─────────────────────────────────────────────── */}
      <div className={styles.branchSection}>
        <div className={styles.branchSectionHeader}>
          <span className={styles.branchSectionLine} />
          <p className={styles.branchSectionTitle}>Branch Analysis</p>
          <span className={styles.branchSectionLine} />
        </div>

        <div className={styles.branchGrid}>
          {Object.entries(branch_scores).map(([key, score], i) => (
            <BranchCard key={key} branchKey={key} score={score} index={i} />
          ))}
        </div>
      </div>
    </section>
  );
}
