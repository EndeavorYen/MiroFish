import { defineConfig } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// The run page against a mocked API (#66): no backend or model service.
const here = path.dirname(fileURLToPath(import.meta.url))
const PORT = 3100

export default defineConfig({
  testDir: here,
  testMatch: /\.spec\.mjs$/,
  timeout: 60_000,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'retain-on-failure',
    locale: 'zh-TW',
    testIdAttribute: 'data-test',
  },
  outputDir: path.join(here, '..', 'artifacts', 'ui-results'),
  webServer: {
    command: `npx vite --port ${PORT} --strictPort --host 127.0.0.1`,
    cwd: path.resolve(here, '../../../frontend'),
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: false,
    timeout: 60_000,
    env: { BROWSER: 'none' },
  },
})
