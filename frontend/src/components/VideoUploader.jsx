/**
 * frontend/src/components/VideoUploader.jsx
 *
 * Drag-and-drop / click-to-upload component for video files.
 * Calls the backend /predict endpoint and passes results upward via onResult.
 */

import { useState, useRef, useCallback } from "react";
import { predictVideo } from "../services/api";
import styles from "./VideoUploader.module.css";

// Accepted MIME types (mirrors backend validation)
const ACCEPTED_TYPES = ["video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"];
const MAX_FILE_SIZE_MB = 200;

export default function VideoUploader({ onResult, onError }) {
  const [isDragging, setIsDragging]   = useState(false);
  const [isLoading, setIsLoading]     = useState(false);
  const [uploadPct, setUploadPct]     = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  // ── File validation ──────────────────────────────────────────────────────
  const validateFile = (file) => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      return `Unsupported file type: ${file.type}. Please upload MP4, AVI, or MOV.`;
    }
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      return `File too large. Maximum size is ${MAX_FILE_SIZE_MB} MB.`;
    }
    return null;
  };

  // ── Core upload logic ────────────────────────────────────────────────────
  const handleUpload = useCallback(async (file) => {
    const validationError = validateFile(file);
    if (validationError) {
      onError?.(validationError);
      return;
    }

    setSelectedFile(file);
    setIsLoading(true);
    setUploadPct(0);
    onResult?.(null); // clear previous result

    try {
      const result = await predictVideo(file, (pct) => setUploadPct(pct));
      onResult?.(result);
    } catch (err) {
      onError?.(err.message || "Prediction failed. Please try again.");
    } finally {
      setIsLoading(false);
      setUploadPct(0);
    }
  }, [onResult, onError]);

  // ── Drag-and-drop handlers ────────────────────────────────────────────────
  const onDragOver  = (e) => { e.preventDefault(); setIsDragging(true);  };
  const onDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
  const onDrop      = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleUpload(file);
  };

  const onFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div
      className={`${styles.dropzone} ${isDragging ? styles.dragging : ""} ${isLoading ? styles.loading : ""}`}
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

      {isLoading ? (
        <div className={styles.progressWrapper}>
          <div className={styles.spinner} aria-label="Processing" />
          <p className={styles.statusText}>
            {uploadPct < 100
              ? `Uploading… ${uploadPct}%`
              : "Analysing video — please wait…"}
          </p>
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
        </div>
      ) : (
        <div className={styles.idleContent}>
          <div className={styles.icon} aria-hidden="true">🎬</div>
          <p className={styles.primaryText}>
            {selectedFile ? `✓ ${selectedFile.name}` : "Drop a video here"}
          </p>
          <p className={styles.secondaryText}>
            or click to browse — MP4, AVI, MOV · max {MAX_FILE_SIZE_MB} MB
          </p>
        </div>
      )}
    </div>
  );
}
