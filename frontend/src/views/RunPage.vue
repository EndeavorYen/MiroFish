<template>
  <div class="run-page">
    <header class="app-header">
      <div class="brand" @click="router.push('/')">MIROFISH</div>
      <div class="header-right">
        <router-link v-if="runId" class="header-link" to="/">{{ $t('run.newRun') }}</router-link>
        <LanguageSwitcher />
      </div>
    </header>

    <main class="content">
      <template v-if="!runId">
        <RunInput @started="(id) => router.push(`/runs/${id}`)" />
        <RunHistory />
      </template>
      <template v-else>
        <p v-if="run && refreshError" class="notice error" data-test="refresh-error">{{ $t('run.refreshFailed', { reason: refreshError }) }}</p>
        <p v-if="loadError" class="notice error" data-test="load-error">{{ loadError }}</p>
        <p v-else-if="!run" class="notice">{{ $t('run.loading') }}</p>
        <RunResult v-else-if="run.status === 'completed'" :run="run" />
        <RunProgress
          v-else
          :run="run"
          :events="events"
          :busy="busy"
          :action-error="actionError"
          @resume="resume"
          @confirm="confirm"
        />
      </template>
    </main>
  </div>
</template>

<script setup>
import { onBeforeUnmount, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import LanguageSwitcher from '../components/LanguageSwitcher.vue'
import RunInput from '../components/run/RunInput.vue'
import RunHistory from '../components/run/RunHistory.vue'
import RunProgress from '../components/run/RunProgress.vue'
import RunResult from '../components/run/RunResult.vue'
import { STOPPED, confirmRun, followRun, getRun, resumeRun } from '../api/runs'

const props = defineProps({ runId: { type: String, default: '' } })
const router = useRouter()

const run = ref(null)
const events = ref([])
const loadError = ref('')
const refreshError = ref('')
const actionError = ref('')
const busy = ref(false)
let lastId = 0
let close = null
let retry = null
// Answers for another run, or older than one already applied, are dropped:
// ``gen`` changes when the page switches run, ``refreshSeq`` on every refresh.
let gen = 0
let refreshSeq = 0

function stopFollowing() {
  close?.()
  close = null
  clearTimeout(retry)
}

/** 'ok', 'stale' (dropped), 'gone' (404) or 'failed'. */
async function refresh() {
  const myGen = gen
  const mySeq = ++refreshSeq
  try {
    const data = (await getRun(props.runId)).data
    if (myGen !== gen || mySeq !== refreshSeq) return 'stale'
    run.value = data
    loadError.value = refreshError.value = ''
    return 'ok'
  } catch (error) {
    if (myGen !== gen) return 'stale'
    // Before anything was shown, the page is an error; after, a notice.
    if (run.value) refreshError.value = error.message
    else loadError.value = error.message
    return error.response?.status === 404 ? 'gone' : 'failed'
  }
}

function follow() {
  stopFollowing()
  const myGen = gen
  close = followRun(props.runId, lastId, {
    onEvent(event) {
      if (myGen !== gen) return
      lastId = event.id
      const last = events.value[events.value.length - 1]
      // Only the latest progress of a stage matters: the log stays short.
      if (event.kind === 'progress' && last?.kind === 'progress' && last.stage === event.stage) {
        events.value[events.value.length - 1] = event
      } else {
        events.value.push(event)
      }
      if (['stage_done', 'run_done', 'run_failed', 'interrupted', 'awaiting_confirmation'].includes(event.kind)) {
        refresh()  // status and artifacts
      }
    },
    async onEnd() {
      if (myGen !== gen) return
      close = null
      const result = await refresh()
      if (myGen !== gen || result === 'gone') return
      // Still going (a network drop, or the backend restarted): follow on
      // from the last event; a stopped run's stream ended on purpose.
      if (run.value && !STOPPED.includes(run.value.status)) {
        retry = setTimeout(follow, 2000)
      }
    },
  })
}

async function load() {
  gen += 1
  stopFollowing()
  run.value = null
  events.value = []
  lastId = 0
  loadError.value = refreshError.value = ''
  if (!props.runId) return
  const myGen = gen
  await refresh()
  if (myGen === gen && run.value && run.value.status !== 'completed') follow()
}

async function act(call) {
  const myGen = gen
  busy.value = true
  actionError.value = ''
  try {
    await call(props.runId)
    await refresh()
    if (myGen === gen) follow()
  } catch (error) {
    actionError.value = error.message
  } finally {
    busy.value = false
  }
}

const resume = () => act(resumeRun)
const confirm = () => act(confirmRun)

watch(() => props.runId, load, { immediate: true })
onBeforeUnmount(stopFollowing)
</script>

<style scoped>
.run-page {
  min-height: 100vh;
  background: #fff;
}

.app-header {
  height: 60px;
  border-bottom: 1px solid #eaeaea;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
}

.brand {
  font-weight: 800;
  font-size: 18px;
  letter-spacing: 1px;
  cursor: pointer;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.header-link {
  color: #000;
  font-size: 13px;
}

.content {
  max-width: 960px;
  margin: 0 auto;
  padding: 32px 24px 64px;
  display: flex;
  flex-direction: column;
  gap: 32px;
}

.notice {
  font-size: 14px;
  color: #555;
}

.notice.error {
  color: #b00020;
}
</style>
