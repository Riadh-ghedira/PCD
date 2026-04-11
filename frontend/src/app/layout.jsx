import "./globals.css";

export const metadata = {
  title: "DeepFake Detector — Multi-Branch Ensemble",
  description:
    "Upload a video and get a deepfake probability score powered by Temporal-Spatial Flow, Audio-Visual Sync, Frequency Analysis, and Biological rPPG.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
