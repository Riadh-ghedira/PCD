/**
 * frontend/src/services/api.js
 *
 * Axios-based service layer that talks to the FastAPI backend.
 * All backend calls route through this module so base URL changes
 * only need to be made here.
 */

import axios from "axios";

// ── Base configuration ────────────────────────────────────────────────────────
const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  process.env.REACT_APP_API_URL ||
  "http://localhost:8000";

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000, // 2-minute timeout for video processing
  headers: {
    Accept: "application/json",
  },
});

// ── Response interceptor (global error handling) ──────────────────────────────
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const detail =
      error.response?.data?.detail ||
      error.message ||
      "An unknown error occurred.";
    return Promise.reject(new Error(detail));
  }
);

// ── API calls ──────────────────────────────────────────────────────────────────

/**
 * Upload a video file for deepfake analysis.
 *
 * @param {File}      file         - The video File object from the input element.
 * @param {Function}  onProgress   - Optional progress callback: (percent: number) => void
 * @returns {Promise<PredictionResult>}
 *
 * @typedef {Object} PredictionResult
 * @property {string}  job_id                - Unique job identifier
 * @property {number}  deepfake_probability  - 0.0 (real) → 1.0 (fake)
 * @property {string}  verdict               - "REAL" | "FAKE" | "UNCERTAIN"
 * @property {Object}  branch_scores         - Per-branch probability scores
 * @property {number}  processing_time_ms    - Server-side processing time
 */
export async function predictVideo(file, onProgress = null) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiClient.post("/predict", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    onUploadProgress: onProgress
      ? (progressEvent) => {
          const percent = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 1)
          );
          onProgress(percent);
        }
      : undefined,
  });

  return response.data;
}

/**
 * Health-check the backend API.
 * @returns {Promise<{status: string, pipeline_ready: boolean}>}
 */
export async function checkHealth() {
  const response = await apiClient.get("/health");
  return response.data;
}

export default apiClient;
