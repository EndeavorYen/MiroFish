import service from './index'

// The runs API (#63): one call runs the whole flow; progress comes as
// Server-Sent Events.

export const STAGES = ['ontology', 'graph', 'prepare', 'simulate', 'report', 'consistency']
// No thread is working on a run in these: the event stream ends.
export const STOPPED = ['completed', 'failed', 'interrupted', 'awaiting_confirmation']

export const createRun = ({ file, text, requirement, maxRounds, seeds, confirmRoles }) => {
  const form = new FormData()
  if (file) form.append('file', file)
  else form.append('document_text', text)
  form.append('simulation_requirement', requirement)
  form.append('max_rounds', String(maxRounds))
  form.append('seeds', String(seeds))
  form.append('confirm_roles', confirmRoles ? 'true' : 'false')
  return service.post('/api/runs', form, { headers: { 'Content-Type': 'multipart/form-data' } })
}

export const getRun = (runId) => service.get(`/api/runs/${runId}`)
export const listRuns = (limit = 100) => service.get('/api/runs', { params: { limit } })
export const resumeRun = (runId) => service.post(`/api/runs/${runId}/resume`)
export const confirmRun = (runId) => service.post(`/api/runs/${runId}/confirm`)

export const getRoles = (graphId) => service.get(`/api/graph/${graphId}/roles`)
export const editRoles = (graphId, ops) => service.post(`/api/graph/${graphId}/roles`, { ops })
export const getGraphData = (graphId) => service.get(`/api/graph/data/${graphId}`)
export const getProfiles = (simulationId) => service.get(`/api/simulation/${simulationId}/profiles`, { params: { platform: 'reddit' } })
export const getPosts = (simulationId, platform, limit = 20) =>
  service.get(`/api/simulation/${simulationId}/posts`, { params: { platform, limit } })
export const getReport = (reportId) => service.get(`/api/report/${reportId}`)
export const getReportMetrics = (reportId) => service.get(`/api/report/${reportId}/metrics`)

/**
 * Follow a run's events after event ``after``. The server ends the stream
 * once the run has stopped; ``onEnd`` is then called (also on a network
 * error) and the caller refetches the run to decide whether to follow again
 * from the last event id. EventSource's own reconnect is not used: it would
 * reopen a finished run's stream every few seconds.
 * Returns a function that closes the stream.
 */
export const followRun = (runId, after, { onEvent, onEnd }) => {
  const base = service.defaults.baseURL || ''
  const source = new EventSource(`${base}/api/runs/${runId}/events?after=${after}`)
  const handle = (message) => onEvent(JSON.parse(message.data))
  for (const kind of ['run_start', 'stage_start', 'progress', 'stage_done', 'run_done', 'run_failed',
    'interrupted', 'awaiting_confirmation']) {
    source.addEventListener(kind, handle)
  }
  source.onerror = () => {
    source.close()
    onEnd()
  }
  return () => source.close()
}
