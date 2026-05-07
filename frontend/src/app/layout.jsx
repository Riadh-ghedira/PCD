import "./globals.css";
import { ThemeProvider } from "../context/ThemeContext";

export const metadata = {
  title: "Deepfake Detection Platform — PCD 2025-2026",
  description:
    "Multi-Branch Ensemble Neural Network for deepfake detection. Four independent analysis modules: Temporal-Spatial Flow, Audio-Visual Sync, Spatial-Frequency Artifacts, and Biological rPPG.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
