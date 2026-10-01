// End-to-end run of the local profile through the run page (#66, #49):
// paste the golden seed, press Start once, and wait for the result page with
// its confidence cards.
//
// Needs the backend with MIROFISH_PROFILE=local on :5001, the frontend on
// :3000, and the local model servers (llama-server :8000, embeddings :8001).
// See docs/local-first.md. Not part of the default CI; the mocked UI tests
// are `npm run test:ui`.
import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

test.use({ testIdAttribute: 'data-test' })

const here = path.dirname(fileURLToPath(import.meta.url))
const fixture = path.resolve(here, '../../backend/tests/fixtures/golden_scenario')
const seedText = fs.readFileSync(path.join(fixture, 'news_seed.txt'), 'utf-8')
const requirement = fs.readFileSync(path.join(fixture, 'simulation_requirement.txt'), 'utf-8').trim()
const ROUNDS = Number(process.env.E2E_ROUNDS || 10)
const SEEDS = Number(process.env.E2E_SEEDS || 2)
const shots = process.env.E2E_SHOTS || path.resolve(here, 'artifacts')

test('local profile: one click on Start reaches the result page', async ({ page }) => {
  fs.mkdirSync(shots, { recursive: true })

  await page.goto('/')
  await page.getByTestId('document-text').fill(seedText)
  await page.getByTestId('requirement').fill(requirement)
  await page.locator('details.advanced summary').click()  // settings only, not a step
  await page.getByTestId('rounds').fill(String(ROUNDS))
  await page.getByTestId('seeds').fill(String(SEEDS))
  await page.screenshot({ path: path.join(shots, '1-input.png') })
  await page.getByTestId('start').click()

  await expect(page).toHaveURL(/\/runs\/run_[0-9a-f]+$/)
  await expect(page.getByTestId('run-status')).toBeVisible()
  await page.screenshot({ path: path.join(shots, '2-progress.png') })

  // The whole flow, with no further clicks.
  await expect(page.getByTestId('run-result')).toBeVisible({ timeout: 60 * 60_000 })
  await expect(page.getByTestId('confidence-cards')).toBeVisible({ timeout: 60_000 })
  await expect(page.getByTestId('card-main-camp')).toContainText(/支持|中立|反對|Support|Neutral|Oppose/)
  if (SEEDS > 1) await expect(page.getByTestId('consistency')).toBeVisible()
  await page.screenshot({ path: path.join(shots, '3-result.png'), fullPage: true })
})
