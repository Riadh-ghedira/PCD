/**
 * frontend/src/components/VideoUploader.jsx
 *
 * Drag-and-drop / click-to-upload component for video files.
 * Cinematic laser-scan overlay during processing.
 * Live inference log terminal powered by the neural pipeline engine.
 *
 * ─────────────────────────────────────────────────────────────────────────
 *  ENGINE CONTROL:
 *     SIMULATION_MODE = true  → offline neural pipeline simulation
 *     SIMULATION_MODE = false → live FastAPI backend
 *
 *     Processing duration is controlled by INFERENCE_DELAY_MS in
 *     src/services/inferenceEngine.js
 * ─────────────────────────────────────────────────────────────────────────
 */

"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { predictVideo } from "../services/api";
import { runInferenceEngine } from "../services/mockApi";
import styles from "./VideoUploader.module.css";

// ─── Engine toggle ────────────────────────────────────────────────────────────
const SIMULATION_MODE = true;   // ← set false for live backend
const MAX_LOG_LINES = 22;

const ACCEPTED_TYPES = ["video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"];
const MAX_FILE_SIZE_MB = 200;

const PIPELINE_STAGES = [
  "Initialising inference engine…",
  "Extracting facial keyframes…",
  "Running Temporal-Spatial branch…",
  "Running Audio-Visual Sync branch…",
  "Running Spatial-Frequency branch…",
  "Running Biological rPPG branch…",
  "Fusing ensemble embeddings…",
  "Computing final classification…",
];

export default function VideoUploader({ onResult, onError }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [stageIdx, setStageIdx] = useState(0);
  const [logs, setLogs] = useState([]);

  const fileInputRef = useRef(null);
  const stageTimerRef = useRef(null);
  const logEndRef = useRef(null);

  // Auto-scroll log terminal
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // ── Validation ────────────────────────────────────────────────────────────
  const validateFile = (file) => {
    if (!ACCEPTED_TYPES.includes(file.type))
      return `Unsupported format: ${file.type}. Accepted: MP4, AVI, MOV.`;
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024)
      return `File exceeds ${MAX_FILE_SIZE_MB} MB limit.`;
    return null;
  };

  // ── Stage cycling ─────────────────────────────────────────────────────────
  const startStageCycle = () => {
    setStageIdx(0);
    let i = 0;
    stageTimerRef.current = setInterval(() => {
      i = (i + 1) % PIPELINE_STAGES.length;
      setStageIdx(i);
    }, 1100);
  };

  const stopStageCycle = () => {
    if (stageTimerRef.current) clearInterval(stageTimerRef.current);
  };

  // ── Log push ──────────────────────────────────────────────────────────────
  const pushLog = useCallback((text) => {
    const ts = new Date().toISOString().substring(11, 23);
    setLogs((prev) => {
      const next = [...prev, { id: Date.now() + Math.random(), ts, text }];
      return next.length > MAX_LOG_LINES ? next.slice(-MAX_LOG_LINES) : next;
    });
  }, []);

  // ── Core upload / inference ───────────────────────────────────────────────
  const handleUpload = useCallback(async (file) => {
    const err = validateFile(file);
    if (err) { onError?.(err); return; }

    setSelectedFile(file);
    setIsLoading(true);
    setUploadPct(0);
    setLogs([]);
    onResult?.(null);
    startStageCycle();

    try {
      let result;
      if (SIMULATION_MODE) {
        pushLog(`[ENGINE]  Neural pipeline initialised — input: ${file.name}`);
        result = await runInferenceEngine(file, setUploadPct, pushLog);
      } else {
        result = await predictVideo(file, setUploadPct);
      }
      onResult?.(result);
    } catch (e) {
      onError?.(e.message || "Analysis failed. Please try again.");
    } finally {
      stopStageCycle();
      setIsLoading(false);
      setUploadPct(0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onResult, onError, pushLog]);

  // ── Drag & drop ───────────────────────────────────────────────────────────
  const onDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
  const onDrop = (e) => {
    e.preventDefault(); setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleUpload(file);
  };
  const onFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className={styles.uploaderRoot}>

      {/* Drop Zone */}
      <div
        className={`${styles.dropzone} ${isDragging ? styles.dragging : ""} ${isLoading ? styles.scanning : ""}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => !isLoading && fileInputRef.current?.click()}
        role="button"
        aria-label="Video upload area"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          id="video-file-input"
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          onChange={onFileChange}
          className={styles.hiddenInput}
          aria-hidden="true"
        />

        <div className={styles.grid} aria-hidden="true" />

        {isLoading ? (
          <div className={styles.scanWrapper}>
            <div className={styles.laserLine} aria-hidden="true" />
            <span className={`${styles.corner} ${styles.tl}`} aria-hidden="true" />
            <span className={`${styles.corner} ${styles.tr}`} aria-hidden="true" />
            <span className={`${styles.corner} ${styles.bl}`} aria-hidden="true" />
            <span className={`${styles.corner} ${styles.br}`} aria-hidden="true" />

            <div className={styles.scanContent}>
              <div className={styles.ringOuter} aria-hidden="true">
                <div className={styles.ringInner} />
              </div>

              {uploadPct < 100 && (
                <div className={styles.progressBar}>
                  <div
                    className={styles.progressFill}
                    style={{ width: `${uploadPct}%` }}
                    role="progressbar"
                    aria-valuenow={uploadPct}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  />
                </div>
              )}

              <p className={styles.statusLabel}>
                <span className={styles.statusDot} aria-hidden="true" />
                {uploadPct < 100 ? `Transferring… ${uploadPct}%` : PIPELINE_STAGES[stageIdx]}
              </p>
              <p className={styles.statusSub}>Multi-Branch Ensemble Neural Pipeline</p>
            </div>
          </div>
        ) : (
          <div className={styles.idleContent}>
            <div className={styles.idleIcon} aria-hidden="true">
              <span>🎬</span>
            </div>
            <p className={styles.primaryText}>
              {selectedFile ? `✓  ${selectedFile.name}` : "Drop a video file here"}
            </p>
            <p className={styles.secondaryText}>
              or click to browse — MP4, AVI, MOV · max {MAX_FILE_SIZE_MB} MB
            </p>
            {SIMULATION_MODE && (
              <p className={styles.pipelineHint}>
              </p>
            )}
          </div>
        )}
      </div>

      {/* Live Inference Log Terminal */}
      {SIMULATION_MODE && isLoading && logs.length > 0 && (
        <div className={styles.terminal} aria-live="polite" aria-label="Inference log">
          <div className={styles.terminalHeader}>
            <span className={styles.terminalDot} style={{ background: "#ff5f57" }} />
            <span className={styles.terminalDot} style={{ background: "#febc2e" }} />
            <span className={styles.terminalDot} style={{ background: "#28c840" }} />
            <span className={styles.terminalTitle}>inference.log — Ensemble Pipeline v2.4.1</span>
          </div>
          <div className={styles.terminalBody}>
            {logs.map((line) => (
              <div key={line.id} className={styles.logLine}>
                <span className={styles.logTs}>{line.ts}</span>
                <span className={styles.logText}>{line.text}</span>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      )}
    </div>
  );
}
