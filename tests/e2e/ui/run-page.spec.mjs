// The one-page run UI (#66) against a mocked API: input -> progress -> result
// with one click, a failed run, and the role confirmation pause.
import { test, expect } from '@playwright/test'

const RUN = 'run_0a1b2c3d4e5f'
const CORS = { 'Access-Control-Allow-Origin': '*' }
const STAGES = ['ontology', 'graph', 'prepare', 'simulate', 'report', 'consistency']

const sse = (events) => events.map((e) => `id: ${e.id}\nevent: ${e.kind}\ndata: ${JSON.stringify(e)}\n\n`).join('')
const event = (id, stage, kind, payload = {}) => ({ id, ts: '2026-10-01T00:00:00', stage, kind, payload })
const json = (data) => ({ status: 200, contentType: 'application/json', body: JSON.stringify({ success: true, data }) })

const runRecord = (status, extra = {}) => ({
  run_id: RUN,
  status,
  stage: extra.stage ?? null,
  error: extra.error ?? null,
  created_at: '2026-10-01T10:00:00+00:00',
  updated_at: '2026-10-01T10:05:00+00:00',
  params: { simulation_requirement: '預測使用者對漲價的反應', max_rounds: 12, seeds: 3, confirm_roles: false },
  artifacts: extra.artifacts ?? {},
})

const DONE_ARTIFACTS = {
  done_stages: STAGES,
  project_id: 'proj_1',
  graph_id: 'mirofish_g1',
  simulation_id: 'sim_1',
  report_id: 'report_abc123',
  rounds: 12,
  seed_simulations: [{ seed: 1, simulation_id: 'sim_1' }, { seed: 1001, simulation_id: 'sim_2' }, { seed: 2001, simulation_id: 'sim_3' }],
  consistency: {
    seeds: 3,
    main_camp: { value: 'support', agree: 3 },
    tendency: { mean: 0.7647, sd: 0.0235 },
    trend: { mean: 0.0151, sd: 0.037 },
    ranking: { most_supportive: ['林志豪'], most_opposed: ['北港工商發展協會'], stability: 0.9, indistinct: false },
    confirm: ['走向的正負在 seed 之間不一致'],
  },
}

const METRICS = {
  scan: {
    main_camp: { value: 'support', counts: { oppose: 2, neutral: 3, support: 6 }, confidence: 'high', evidence: '本機路徑在評估庫 6/6 個情境與 LLM 路徑一致' },
    tendency: { value: 0.76, confidence: 'medium', evidence: '本機路徑在評估庫 3/6 個情境與 LLM 路徑一致' },
    trend: { value: 0.05, rounds: [0, 11], confidence: 'low', evidence: '本機路徑在評估庫 3/6 個情境與 LLM 路徑一致' },
    ranking: {
      most_supportive: [{ name: '林志豪', stance: 0.9 }], most_opposed: [{ name: '北港工商發展協會', stance: 0.2 }],
      indistinct: false, confidence: 'low', evidence: '本機路徑在評估庫 2/6 個情境與 LLM 路徑一致',
    },
  },
  spread: { twitter: [{ author: '數位前線', content: '水費調漲三成，管線汰換真的需要這麼多嗎？', reposts_and_quotes: 4, likes: 9 }] },
}

/** Every API call the page makes, answered from ``state``; unknown ones fail the test. */
async function mockApi(page, state) {
  state.requests = []
  // Only the backend's /api paths: Vite serves the app's own src/api/*.js too.
  await page.route((url) => url.pathname.startsWith('/api/'), async (route) => {
    const request = route.request()
    // The page calls the backend on :5001 directly: a cross-origin request.
    if (request.method() === 'OPTIONS') {
      return route.fulfill({ status: 204, headers: { ...CORS, 'Access-Control-Allow-Methods': 'GET, POST', 'Access-Control-Allow-Headers': 'Content-Type, Accept-Language' } })
    }
    const url = new URL(request.url())
    const key = `${request.method()} ${url.pathname}`
    state.requests.push({ key, body: request.postData() })
    const handler = state.routes[key]
    if (!handler) return route.fulfill({ status: 500, headers: CORS, body: `unmocked ${key}` })
    const answer = typeof handler === 'function' ? handler(request) : handler
    if (answer.abort) return route.abort(answer.abort)
    if (answer.sse) {
      return route.fulfill({ status: 200, contentType: 'text/event-stream', headers: CORS, body: sse(answer.sse) })
    }
    return route.fulfill({ ...answer, headers: { ...CORS, ...(answer.headers || {}) } })
  })
}

test('input to result with one click on Start', async ({ page }) => {
  let started = false
  const state = {
    routes: {
      'GET /api/runs': json([]),
      'POST /api/runs': () => { started = true; return { status: 202, contentType: 'application/json', body: JSON.stringify({ success: true, data: { run_id: RUN } }) } },
      // Running until the stream has been read; then the finished run.
      [`GET /api/runs/${RUN}`]: () => json(state.streamed ? runRecord('completed', { artifacts: DONE_ARTIFACTS }) : runRecord('running', { stage: 'ontology' })),
      [`GET /api/runs/${RUN}/events`]: () => {
        state.streamed = true
        let id = 0
        return {
          sse: [
            event(++id, null, 'run_start', { resume_from: 'ontology' }),
            ...STAGES.flatMap((stage) => [event(++id, stage, 'stage_start'), event(++id, stage, 'stage_done', { artifacts: {}, seconds: 1.2 })]),
            event(++id, null, 'run_done', { artifacts: DONE_ARTIFACTS }),
          ],
        }
      },
      'GET /api/report/report_abc123/metrics': json(METRICS),
      'GET /api/report/report_abc123': json({ markdown_content: '# 模擬指標報告\n\n摘要。' }),
    },
  }
  await mockApi(page, state)

  await page.goto('/')
  await page.getByTestId('document-text').fill('北港市擬調漲自來水費三成。')
  await page.getByTestId('requirement').fill('預測使用者對漲價的反應')
  await page.getByTestId('start').click()  // the only click

  await expect(page).toHaveURL(new RegExp(`/runs/${RUN}$`))
  await expect(page.getByTestId('run-result')).toBeVisible()
  const cards = page.getByTestId('confidence-cards')
  await expect(cards.getByTestId('card-main-camp')).toContainText('支持')
  await expect(cards.getByTestId('card-main-camp')).toContainText('可信度高')
  await expect(cards.getByTestId('card-ranking')).toContainText('林志豪')
  await expect(page.getByTestId('consistency')).toContainText('3/3')
  await expect(page.getByTestId('consistency-confirm')).toContainText('走向的正負')
  await expect(page.getByTestId('top-posts')).toContainText('水費調漲三成')

  expect(started).toBe(true)
  const form = state.requests.find((r) => r.key === 'POST /api/runs').body
  expect(form).toContain('預測使用者對漲價的反應')
  expect(form).toContain('name="seeds"')
})

test('a failed run shows its reason and resumes', async ({ page }) => {
  const reason = '模型服務無回應：http://127.0.0.1:8000/v1（ConnectError）'
  const failed = runRecord('failed', { stage: 'simulate', error: reason, artifacts: { done_stages: STAGES.slice(0, 3) } })
  const state = {
    routes: {
      [`GET /api/runs/${RUN}`]: () => json(state.resumed ? runRecord('running', { stage: 'simulate', artifacts: failed.artifacts }) : failed),
      [`GET /api/runs/${RUN}/events`]: { sse: [event(1, 'simulate', 'stage_start'), event(2, 'simulate', 'run_failed', { reason })] },
      [`POST /api/runs/${RUN}/resume`]: () => { state.resumed = true; return { status: 202, contentType: 'application/json', body: JSON.stringify({ success: true, data: { run_id: RUN } }) } },
    },
  }
  await mockApi(page, state)

  await page.goto(`/runs/${RUN}`)
  await expect(page.getByTestId('run-problem')).toBeVisible()
  await expect(page.getByTestId('run-problem')).toContainText('模擬互動')  // the stage it stopped at
  await expect(page.getByTestId('run-error')).toHaveText(reason)
  await expect(page.locator('[data-stage="simulate"]')).toHaveClass(/failed/)
  await expect(page.locator('[data-stage="prepare"]')).toHaveClass(/done/)

  await page.getByTestId('resume').click()
  await expect.poll(() => state.requests.some((r) => r.key === `POST /api/runs/${RUN}/resume`)).toBe(true)
  await expect(page.getByTestId('run-status')).toHaveText('進行中')
})

test('a paused run lets the user drop a role and continue', async ({ page }) => {
  const paused = runRecord('awaiting_confirmation', { stage: 'prepare', artifacts: { done_stages: ['ontology', 'graph'], graph_id: 'mirofish_g1' } })
  const roles = [
    { uuid: 'u1', name: '鄭國棟', type: 'GovernmentOfficial', summary: '事業處處長', excluded: false, aliases: [] },
    { uuid: 'u2', name: '北港', type: 'Location', summary: '地名，不是角色', excluded: false, aliases: [] },
  ]
  const state = {
    routes: {
      [`GET /api/runs/${RUN}`]: () => json(state.confirmed ? runRecord('running', { stage: 'prepare', artifacts: paused.artifacts }) : paused),
      [`GET /api/runs/${RUN}/events`]: { sse: [event(1, 'prepare', 'awaiting_confirmation', { graph_id: 'mirofish_g1' })] },
      'GET /api/graph/mirofish_g1/roles': json({ graph_id: 'mirofish_g1', roles }),
      'POST /api/graph/mirofish_g1/roles': json({ graph_id: 'mirofish_g1', roles }),
      [`POST /api/runs/${RUN}/confirm`]: () => { state.confirmed = true; return { status: 202, contentType: 'application/json', body: JSON.stringify({ success: true, data: { run_id: RUN } }) } },
    },
  }
  await mockApi(page, state)

  await page.goto(`/runs/${RUN}`)
  const panel = page.getByTestId('roles-confirm')
  await expect(panel.locator('input.name').first()).toHaveValue('鄭國棟')  // names are editable
  await panel.locator('li', { hasText: '地名' }).locator('input[type="checkbox"]').uncheck()
  await page.getByTestId('confirm-roles').click()

  await expect.poll(() => state.requests.some((r) => r.key === `POST /api/runs/${RUN}/confirm`)).toBe(true)
  const edit = state.requests.find((r) => r.key === 'POST /api/graph/mirofish_g1/roles')
  expect(JSON.parse(edit.body)).toEqual({ ops: [{ op: 'exclude', uuid: 'u2' }] })
})

test('a reset connection is retried instead of leaving a stale page', async ({ page }) => {
  // The dev server's /api proxy sometimes resets a connection (#66).
  let attempts = 0
  const state = {
    routes: {
      [`GET /api/runs/${RUN}`]: () => (++attempts <= 2 ? { abort: 'connectionreset' } : json(runRecord('completed', { artifacts: DONE_ARTIFACTS }))),
      'GET /api/report/report_abc123/metrics': json(METRICS),
      'GET /api/report/report_abc123': { abort: 'connectionreset' },  // the report text is optional
    },
  }
  await mockApi(page, state)

  await page.goto(`/runs/${RUN}`)
  await expect(page.getByTestId('card-main-camp')).toContainText('支持')
  expect(attempts).toBe(3)
})
