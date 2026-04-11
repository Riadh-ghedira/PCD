/**
 * frontend/src/components/ScoreDisplay.jsx
 *
 * Renders the ensemble deepfake probability score, the verdict badge,
 * and the per-branch breakdown sent back by the backend.
 */

import styles from "./ScoreDisplay.module.css";

// Colour coding by verdict
const VERDICT_META = {
  REAL:      { colour: "#22c55e", emoji: "✅", label: "Likely Real"    },
  FAKE:      { colour: "#ef4444", emoji: "🚨", label: "Likely Deepfake" },
  UNCERTAIN: { colour: "#f59e0b", emoji: "⚠️",  label: "Uncertain"      },
};

const BRANCH_LABELS = {
  Temporal_Spatial_Flow:       "Temporal & Spatial Flow",
  Audio_Visual_Sync:           "Audio-Visual Sync",
  Spatial_Frequency_Artifacts: "Spatial Frequency Artefacts",
  Biological_rPPG:             "Biological rPPG (Pulse)",
};

export default function ScoreDisplay({ result }) {
  if (!result) return null;

  const { deepfake_probability, verdict, branch_scores, processing_time_ms, job_id } = result;
  const meta   = VERDICT_META[verdict] ?? VERDICT_META.UNCERTAIN;
  const pct    = Math.round(deepfake_probability * 100);
  const isHigh = deepfake_probability >= 0.65;

  return (
    <section className={styles.card} aria-live="polite">
      {/* ── Main Score ──────────────────────────────────────────────── */}
      <div className={styles.scoreHeader}>
        <span className={styles.verdictEmoji} role="img" aria-label={verdict}>
          {meta.emoji}
        </span>
        <div>
          <h2 className={styles.verdictLabel} style={{ color: meta.colour }}>
            {meta.label}
          </h2>
          <p className={styles.jobId}>Job ID: {job_id}</p>
        </div>
      </div>

      {/* ── Probability Gauge ────────────────────────────────────────── */}
      <div className={styles.gaugeWrapper}>
        <div className={styles.gaugeTrack}>
          <div
            className={styles.gaugeFill}
            style={{
              width: `${pct}%`,
              backgroundColor: meta.colour,
            }}
            role="progressbar"
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Deepfake probability"
          />
        </div>
        <span className={styles.gaugeLabel} style={{ color: meta.colour }}>
          {pct}%
        </span>
      </div>
      <p className={styles.gaugeCaption}>
        Ensemble Deepfake Probability&nbsp;
        <span className={isHigh ? styles.high : styles.low}>
          ({isHigh ? "High" : "Low"} confidence deepfake signal)
        </span>
      </p>

      {/* ── Per-Branch Breakdown ─────────────────────────────────────── */}
      <div className={styles.branches}>
        <h3 className={styles.branchesTitle}>Branch Breakdown</h3>
        {Object.entries(branch_scores).map(([key, score]) => {
          const branchPct = Math.round(score * 100);
          return (
            <div key={key} className={styles.branchRow}>
              <span className={styles.branchName}>
                {BRANCH_LABELS[key] ?? key}
              </span>
              <div className={styles.branchTrack}>
                <div
                  className={styles.branchFill}
                  style={{
                    width: `${branchPct}%`,
                    backgroundColor: score >= 0.65 ? "#ef4444" : score >= 0.35 ? "#f59e0b" : "#22c55e",
                  }}
                />
              </div>
              <span className={styles.branchScore}>{branchPct}%</span>
            </div>
          );
        })}
      </div>

      {/* ── Footer ───────────────────────────────────────────────────── */}
      <p className={styles.footer}>
        Processed in {processing_time_ms.toLocaleString()} ms
      </p>
    </section>
  );
}
