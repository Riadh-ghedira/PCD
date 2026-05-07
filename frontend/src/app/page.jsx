/**
 * frontend/src/app/page.jsx
 * Main page — wires VideoUploader → ScoreDisplay.
 * Branding: PCD 2025-2026. No placeholder or test text in the UI.
 */

"use client";

import { useState } from "react";
import VideoUploader from "../components/VideoUploader";
import ScoreDisplay  from "../components/ScoreDisplay";
import ThemeToggle   from "../components/ThemeToggle";
import styles from "./page.module.css";

const MODULE_INFO = [
  { icon: "🎞️", label: "Temporal-Spatial Flow",      desc: "3D-CNN · R3D-18"      },
  { icon: "🔊", label: "Audio-Visual Sync",           desc: "Dual-Encoder SyncNet" },
  { icon: "📡", label: "Spatial-Frequency Artifacts", desc: "FFT + ResNet-18"      },
  { icon: "💓", label: "Biological rPPG",             desc: "LSTM Pulse Analysis"  },
];

export default function HomePage() {
  const [result, setResult] = useState(null);
  const [error,  setError]  = useState(null);

  const handleResult = (data) => { setError(null); setResult(data); };
  const handleError  = (msg)  => { setResult(null); setError(msg);  };
  const reset        = ()     => { setResult(null); setError(null); };

  return (
    <div className={styles.page}>
      {/* Ambient background blobs */}
      <div className={styles.blob1} aria-hidden="true" />
      <div className={styles.blob2} aria-hidden="true" />

      {/* Theme toggle — fixed top-right */}
      <ThemeToggle />

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className={styles.header}>
        <div className={styles.headerBadge}>
          <span className={styles.badgeDot} />
          INFERENCE ENGINE ONLINE
        </div>
        <h1 className={styles.title}>
          <span className={styles.titleAccent}>Deepfake</span>
          {" "}Detection Platform
        </h1>
        <p className={styles.subtitle}>
          Multi-Branch Ensemble Neural Network — 4-Module Analysis Pipeline
        </p>
      </header>

      {/* ── Main content ─────────────────────────────────────────────────── */}
      <main className={styles.main}>
        {!result && (
          <div className={styles.uploaderWrapper}>
            <VideoUploader onResult={handleResult} onError={handleError} />
            {error && (
              <p className={styles.errorBanner} role="alert">
                <span aria-hidden="true">⚠</span> {error}
              </p>
            )}
          </div>
        )}

        {result && (
          <div className={styles.resultsWrapper}>
            <ScoreDisplay result={result} />
            <button id="scan-new-btn" className={styles.resetBtn} onClick={reset}>
              ↺ &nbsp;Analyse New Video
            </button>
          </div>
        )}
      </main>

      {/* ── Module strip ─────────────────────────────────────────────────── */}
      <section className={styles.moduleStrip} aria-label="Detection modules">
        {MODULE_INFO.map((m) => (
          <div key={m.label} className={styles.moduleCard}>
            <span className={styles.moduleIcon}>{m.icon}</span>
            <p className={styles.moduleLabel}>{m.label}</p>
            <p className={styles.moduleSub}>{m.desc}</p>
          </div>
        ))}
      </section>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className={styles.footer}>
        <span className={styles.footerDot} />
        PyTorch · FastAPI · Next.js
        <span className={styles.footerSep}>|</span>
        PCD 2025-2026
      </footer>
    </div>
  );
}
