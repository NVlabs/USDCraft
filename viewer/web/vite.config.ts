// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// This file is derived from Articraft (https://github.com/mattzh72/articraft),
// copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
// and has been modified by NVIDIA.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const apiHost = process.env.USDCRAFT_VIEWER_API_HOST ?? "127.0.0.1";
const apiPort = process.env.USDCRAFT_VIEWER_API_PORT ?? "8770";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    chunkSizeWarningLimit: 900,
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL("./index.html", import.meta.url)),
        // Headless review renderer page (A2+): same scene/loader/material code,
        // no UI chrome, driven via window.usdcraftRenderer by Playwright.
        render: fileURLToPath(new URL("./render.html", import.meta.url)),
      },
      output: {
        manualChunks: {
          three: ["three"],
          ui: ["lucide-react", "@radix-ui/react-tabs", "class-variance-authority", "clsx", "tailwind-merge"],
        },
      },
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": `http://${apiHost}:${apiPort}`,
      "/health": `http://${apiHost}:${apiPort}`,
    },
  },
});
