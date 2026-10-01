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
const actionError = ref('')
const busy = ref(false)
let lastId = 0
let close = null
let retry = null

function stopFollowing() {
  close?.()
  close = null
  clearTimeout(retry)
}

async function refresh() {
  try {
    run.value = (await getRun(props.runId)).data
    loadError.value = ''
  } catch (error) {
    loadError.value = error.message
  }
}

function follow() {
  stopFollowing()
  close = followRun(props.runId, lastId, {
    onEvent(event) {
      lastId = event.id
      events.value.push(event)
      if (['stage_done', 'run_done', 'run_failed', 'interrupted', 'awaiting_confirmation'].includes(event.kind)) {
        refresh()  // status and artifacts
      }
    },
    async onEnd() {
      close = null
      await refresh()
      // Still going (a network drop, or the backend restarted): follow on
      // from the last event; a stopped run's stream ended on purpose.
      if (run.value && !STOPPED.includes(run.value.status)) {
        retry = setTimeout(follow, 2000)
      }
    },
  })
}

async function load() {
  stopFollowing()
  run.value = null
  events.value = []
  lastId = 0
  loadError.value = ''
  if (!props.runId) return
  await refresh()
  if (run.value && run.value.status !== 'completed') follow()
}

async function act(call) {
  busy.value = true
  actionError.value = ''
  try {
    await call(props.runId)
    await refresh()
    follow()
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
