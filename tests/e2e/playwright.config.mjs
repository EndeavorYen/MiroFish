import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: '.',
  testMatch: /\.spec\.mjs$/,
  timeout: 90 * 60_000,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://localhost:3000',
    trace: 'retain-on-failure',
  },
  outputDir: 'artifacts/test-results',
})
