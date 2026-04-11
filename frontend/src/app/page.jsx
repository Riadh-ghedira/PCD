/**
 * frontend/src/app/page.jsx  (Next.js App Router)
 *  — or rename to src/pages/index.jsx for the Pages Router.
 *
 * Main page: video upload → deepfake prediction → results display.
 */

"use client";

import { useState, useRef, useCallback } from "react";
import axios from "axios";
import styles from "./page.module.css";

const API_URL = "http://localhost:8000";

const MODULE_INFO = {
  Temporal_Spatial_Flow: { icon: "🎞️", label: "Temporal-Spatial Flow", desc: "3D-CNN / R3D-18" },
  Audio_Visual_Sync: { icon: "🔊", label: "Audio-Visual Sync", desc: "Dual-Encoder SyncNet" },
  Spatial_Frequency_Artifacts: { icon: "📡", label: "Spatial Frequency", desc: "FFT + ResNet-18" },
  Biological_rPPG: { icon: "💓", label: "Biological rPPG", desc: "LSTM-based Pulse" },
};

export default function HomePage() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);

  const handleFile = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type.startsWith("video/")) {
      setFile(selectedFile);
      setError(null);
      setResult(null);
    } else {
      setError("Please upload a valid video file.");
    }
  };

  const uploadVideo = async () => {
    if (!file) return;

    setLoading(true);
    setResult(null);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        },
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred during analysis.");
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>
          <span className={styles.highlight}>Deepfake</span> Detection System
        </h1>
        <p className={styles.subtitle}>Ensemble Multi-Branch Analysis — AI Final Project</p>
      </header>

      <main className={styles.main}>
        {!result && (
          <div className={styles.uploadArea}>
            <div 
              className={styles.dropZone}
              onClick={() => fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFile} 
                className={styles.fileInput} 
                accept="video/*"
              />
              <div className={styles.dropZoneContent}>
                <span className={styles.icon}>📁</span>
                <p>{file ? file.name : "Drag and drop video or click to upload"}</p>
              </div>
            </div>
            
            {file && !loading && (
              <button className={styles.actionButton} onClick={uploadVideo}>
                Start Deepfake Analysis
              </button>
            )}

            {loading && (
              <div className={styles.loadingContainer}>
                <div className={styles.spinner}></div>
                <p>{uploadProgress < 100 ? `Uploading: ${uploadProgress}%` : "Running 4 AI Analysis Branches..."}</p>
              </div>
            )}

            {error && <p className={styles.errorText}>{error}</p>}
          </div>
        )}

        {result && (
          <div className={styles.resultContainer}>
            <div className={styles.scoreOverview}>
              <div 
                className={styles.scoreCircle} 
                style={{ "--percent": String(result.deepfake_probability * 100) }}
              >
                <div className={styles.scoreValue}>
                  {Math.round(result.deepfake_probability * 100)}%
                </div>
              </div>
              <div className={styles.verdict}>
                <h2 className={`${styles.verdictTitle} ${result.verdict === "FAKE" ? styles.fake : styles.real}`}>
                  {result.verdict}
                </h2>
                <p>Final Deepfake Probability</p>
              </div>
            </div>

            <div className={styles.breakdown}>
              <h3 className={styles.breakdownTitle}>Branch Analysis</h3>
              <div className={styles.branchGrid}>
                {Object.entries(result.branch_scores).map(([key, score]) => (
                  <div key={key} className={styles.branchCard}>
                    <div className={styles.branchHeader}>
                      <span className={styles.branchIcon}>{MODULE_INFO[key]?.icon}</span>
                      <span className={styles.branchLabel}>{MODULE_INFO[key]?.label}</span>
                    </div>
                    <div className={styles.branchScore}>
                      <div className={styles.barContainer}>
                        <div 
                          className={styles.barFill} 
                          style={{ width: `${score * 100}%` }}
                        ></div>
                      </div>
                      <span>{Math.round(score * 100)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <button className={styles.secondaryButton} onClick={reset}>
              Scan New Video
            </button>
          </div>
        )}
      </main>

      <section className={styles.infoStrip}>
        {Object.values(MODULE_INFO).map((mod) => (
          <div key={mod.label} className={styles.infoBox}>
            <span className={styles.infoIcon}>{mod.icon}</span>
            <h4>{mod.label}</h4>
            <p>{mod.desc}</p>
          </div>
        ))}
      </section>

      <footer className={styles.footer}>
        <p>Built with PyTorch & FastAPI Backend · Next.js Frontend</p>
      </footer>
    </div>
  );
}


