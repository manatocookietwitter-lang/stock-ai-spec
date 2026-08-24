import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/global-setup.ts",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: "line",
  use: {
    baseURL: "http://127.0.0.1:8766",
    browserName: "chromium",
    channel: "msedge",
    headless: true,
    viewport: { width: 390, height: 844 },
  },
});
