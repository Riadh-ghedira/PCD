/**
 * frontend/src/components/ThemeToggle.jsx
 *
 * Sun/Moon button in the top-right corner of the header.
 * Uses lucide-react for crisp SVG icons.
 */

"use client";

import { Sun, Moon } from "lucide-react";
import { useTheme } from "../context/ThemeContext";
import styles from "./ThemeToggle.module.css";

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  const isLight = theme === "light";

  return (
    <button
      id="theme-toggle-btn"
      className={styles.btn}
      onClick={toggleTheme}
      aria-label={isLight ? "Switch to dark mode" : "Switch to light mode"}
      title={isLight ? "Dark mode" : "Light mode"}
    >
      {isLight ? (
        <Moon size={18} strokeWidth={1.8} />
      ) : (
        <Sun size={18} strokeWidth={1.8} />
      )}
    </button>
  );
}
