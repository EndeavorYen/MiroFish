// End-to-end run of the local profile (#49): upload the golden seed, build the
// graph, prepare, simulate a few rounds, and wait for the metrics report.
//
// Needs the backend with MIROFISH_PROFILE=local on :5001, the frontend on
// :3000, and the local model servers (llama-server :8000, embeddings :8001).
// See docs/local-first.md. Not part of the default CI.
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const fixture = path.resolve(here, '../../backend/tests/fixtures/golden_scenario')
const seedFile = path.join(fixture, 'news_seed.txt')
const requirement = fs.readFileSync(path.join(fixture, 'simulation_requirement.txt'), 'utf-8').trim()
const ROUNDS = Number(process.env.E2E_ROUNDS || 10)
const shots = process.env.E2E_SHOTS || path.resolve(here, 'artifacts')

test('local profile reaches a finished metrics report', async ({ page }) => {
  fs.mkdirSync(shots, { recursive: true })

  // Home: seed file and requirement.
  await page.goto('/')
  await page.locator('input[type="file"]').setInputFiles(seedFile)
  await page.locator('textarea').first().fill(requirement)
  await page.getByRole('button', { name: /启动引擎|Launch|啟動/ }).click()

  // Step 1: ontology and graph, then create the simulation.
  await expect(page).toHaveURL(/\/process\//)
  const enterEnv = page.locator('.action-btn', { hasText: /➝/ }).first()
  await expect(enterEnv).toBeEnabled({ timeout: 15 * 60_000 })
  await page.screenshot({ path: path.join(shots, '1-graph.png') })
  await enterEnv.click()

  // Step 2: prepare, then a short custom run.
  await expect(page).toHaveURL(/\/simulation\/[^/]+$/, { timeout: 60_000 })
  const start = page.locator('.action-btn.primary')
  await expect(start).toBeEnabled({ timeout: 20 * 60_000 })
  await page.locator('input[type="checkbox"]').first().check()
  await page.locator('input[type="range"]').first().fill(String(ROUNDS))
  await expect(page.locator('.val-num').first()).toHaveText(String(ROUNDS))
  await page.screenshot({ path: path.join(shots, '2-env.png') })
  await start.click()

  // Step 3: simulate, then ask for the report.
  await expect(page).toHaveURL(/\/simulation\/[^/]+\/start/, { timeout: 60_000 })
  const report = page.locator('.action-controls .action-btn.primary')
  await expect(report).toBeEnabled({ timeout: 30 * 60_000 })
  await page.screenshot({ path: path.join(shots, '3-sim.png') })
  await report.click()

  // Step 4: the metrics report finishes and polling stops.
  await expect(page).toHaveURL(/\/report\//, { timeout: 60_000 })
  await expect(page.locator('.next-step-btn')).toBeVisible({ timeout: 10 * 60_000 })
  // Every planned section has its generated text.
  const titles = page.locator('.report-section-item .section-title')
  const count = await titles.count()
  expect(count).toBeGreaterThan(0)
  await expect(page.locator('.report-section-item .generated-content')).toHaveCount(count)
  for (const text of await page.locator('.generated-content').allInnerTexts()) {
    expect(text.trim().length).toBeGreaterThan(20)
  }
  await page.screenshot({ path: path.join(shots, '4-report.png'), fullPage: true })

  let polls = 0
  page.on('request', (request) => {
    if (/\/api\/report\/.*(agent-log|console-log)/.test(request.url())) polls += 1
  })
  await page.waitForTimeout(8_000)
  expect(polls).toBe(0)
})
