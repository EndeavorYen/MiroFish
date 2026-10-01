import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: '.',
  testMatch: /\.spec\.mjs$/,
  testIgnore: /[\\/]ui[\\/]/,  // the mocked UI tests have their own config (ui/)
  timeout: 90 * 60_000,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://localhost:3000',
    trace: 'retain-on-failure',
    // A missing element fails in a minute; the long waits have their own timeouts.
    actionTimeout: 60_000,
  },
  outputDir: 'artifacts/test-results',
})
