import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import cssInjectedByJsPlugin from "vite-plugin-css-injected-by-js";

// Builds one self-contained classic IIFE script (a lib build) with the CSS injected by JS.
export default defineConfig({
  plugins: [react(), cssInjectedByJsPlugin()],
  // React and some dependencies read process.env.NODE_ENV at runtime, which is undefined in a plain browser script,
  // so inline it as a literal.
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    lib: {
      entry: "src/main.jsx",
      name: "OpenPointUI",
      formats: ["iife"],
      fileName: () => "open-point-ui.js",
    },
    rollupOptions: {
      // No externals: the bundle must run with only a <script> tag. No import map, no CDN, no network.
      output: { inlineDynamicImports: true },
    },
  },
});
