/**
 * frontend/src/context/ThemeContext.jsx
 *
 * Provides a dark/light mode toggle via React context.
 * Stores preference in localStorage and applies a "light" class to <html>.
 * CSS variables in globals.css handle the actual color overrides.
 */

"use client";

import { createContext, useContext, useEffect, useState } from "react";

const ThemeContext = createContext({ theme: "dark", toggleTheme: () => {} });

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState("dark");

  // On mount, read saved preference
  useEffect(() => {
    const saved = localStorage.getItem("df-theme") ?? "dark";
    setTheme(saved);
    document.documentElement.classList.toggle("light", saved === "light");
  }, []);

  const toggleTheme = () => {
    setTheme((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      localStorage.setItem("df-theme", next);
      document.documentElement.classList.toggle("light", next === "light");
      return next;
    });
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeContext);
