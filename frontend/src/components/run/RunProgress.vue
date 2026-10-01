<template>
  <section class="run-progress" :data-status="run.status">
    <div class="head">
      <h1 class="title">{{ run.params?.simulation_requirement }}</h1>
      <span class="status" :class="run.status" data-test="run-status">{{ $t(`run.status.${run.status}`) }}</span>
    </div>

    <div class="bar" :aria-valuenow="percent" role="progressbar" aria-valuemin="0" aria-valuemax="100">
      <div class="fill" :class="{ failed: stopped }" :style="{ width: `${percent}%` }" />
    </div>

    <!-- Failed or cut off: the reason, and continue where it stopped. -->
    <div v-if="run.status === 'failed' || run.status === 'interrupted'" class="problem" data-test="run-problem">
      <strong>{{ run.status === 'failed' ? $t('run.progress.failedAt', { stage: stageName(run.stage) }) : $t('run.progress.interrupted') }}</strong>
      <p v-if="run.error" class="reason" data-test="run-error">{{ run.error }}</p>
      <p class="hint">{{ $t('run.progress.resumeHint') }}</p>
      <button class="primary" :disabled="busy" data-test="resume" @click="$emit('resume')">{{ $t('run.progress.resume') }}</button>
    </div>
    <p v-if="actionError" class="action-error" data-test="action-error">{{ actionError }}</p>

    <RolesConfirm
      v-if="run.status === 'awaiting_confirmation'"
      :graph-id="run.artifacts?.graph_id"
      :busy="busy"
      @confirm="$emit('confirm')"
    />

    <ol class="stages">
      <li v-for="stage in stages" :key="stage.code" :class="stage.state" :data-stage="stage.code">
        <span class="mark">{{ { done: '✓', running: '…', failed: '✕', pending: '·' }[stage.state] }}</span>
        <span class="name">{{ stageName(stage.code) }}</span>
        <span class="detail">
          <template v-if="stage.state === 'done' && stage.seconds != null">{{ $t('run.progress.seconds', { s: stage.seconds }) }}</template>
          <template v-else-if="stage.message">{{ stage.message }}</template>
        </span>
      </li>
    </ol>

    <div class="peek">
      <details v-if="run.artifacts?.graph_id" @toggle="(e) => e.target.open && loadGraph()">
        <summary>{{ $t('run.progress.showGraph') }}</summary>
        <div class="graph-box">
          <GraphPanel :graph-data="graph" :loading="graphLoading" :current-phase="2" @refresh="loadGraph" />
        </div>
      </details>
      <details v-if="firstSimulation" @toggle="(e) => e.target.open && loadProfiles()">
        <summary>{{ $t('run.progress.showPersonas') }}</summary>
        <ul class="personas">
          <li v-for="(profile, i) in profiles" :key="i">
            <strong>{{ profile.name || profile.username }}</strong>
            <span>{{ (profile.bio || profile.persona || '').slice(0, 140) }}</span>
          </li>
        </ul>
      </details>
      <details v-if="firstSimulation && reachedSimulate" @toggle="togglePosts">
        <summary>{{ $t('run.progress.showPosts') }}</summary>
        <ul class="posts">
          <li v-for="post in posts" :key="post.post_id">{{ post.content }}</li>
          <li v-if="!posts.length" class="empty">{{ $t('run.progress.noPostsYet') }}</li>
        </ul>
      </details>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import GraphPanel from '../GraphPanel.vue'
import RolesConfirm from './RolesConfirm.vue'
import { STAGES, getGraphData, getPosts, getProfiles } from '../../api/runs'

const props = defineProps({
  run: { type: Object, required: true },
  events: { type: Array, required: true },
  busy: Boolean,
  actionError: { type: String, default: '' },
})
defineEmits(['resume', 'confirm'])
const { t } = useI18n()

const stageName = (code) => (code ? t(`run.stages.${code}`) : '')
const stopped = computed(() => ['failed', 'interrupted'].includes(props.run.status))

// Each stage's state from the event log; a resume starts the stage again.
const stages = computed(() => {
  const done = new Set(props.run.artifacts?.done_stages || [])
  const byCode = Object.fromEntries(STAGES.map((code) => [code, {
    code, state: done.has(code) ? 'done' : 'pending', message: '', progress: 0, seconds: null,
  }]))
  for (const event of props.events) {
    const stage = byCode[event.stage]
    if (!stage) continue
    if (event.kind === 'stage_start') Object.assign(stage, { state: 'running', message: '', progress: 0 })
    else if (event.kind === 'progress') {
      if (event.payload.message) stage.message = event.payload.message
      if (typeof event.payload.progress === 'number') stage.progress = event.payload.progress
    } else if (event.kind === 'stage_done') Object.assign(stage, { state: 'done', seconds: event.payload.seconds })
    else if (event.kind === 'run_failed') stage.state = 'failed'
  }
  for (const stage of Object.values(byCode)) {
    if (done.has(stage.code)) stage.state = 'done'
    else if (stage.state === 'running' && stopped.value) stage.state = 'failed'
  }
  return STAGES.map((code) => byCode[code])
})

const percent = computed(() => {
  const list = stages.value
  const done = list.filter((s) => s.state === 'done').length
  const running = list.find((s) => s.state === 'running')
  return Math.round(100 * (done + (running ? Math.min(running.progress, 99) / 100 : 0)) / list.length)
})

const firstSimulation = computed(() => props.run.artifacts?.simulation_id)
const reachedSimulate = computed(() => stages.value.find((s) => s.code === 'simulate').state !== 'pending')

const graph = ref(null)
const graphLoading = ref(false)
async function loadGraph() {
  graphLoading.value = true
  try {
    graph.value = (await getGraphData(props.run.artifacts.graph_id)).data
  } catch {
    graph.value = null
  } finally {
    graphLoading.value = false
  }
}

const profiles = ref([])
async function loadProfiles() {
  try {
    profiles.value = (await getProfiles(firstSimulation.value)).data.profiles || []
  } catch {
    profiles.value = []
  }
}

const posts = ref([])
let postTimer = null
async function loadPosts() {
  try {
    const [twitter, reddit] = await Promise.all(['twitter', 'reddit'].map((p) => getPosts(firstSimulation.value, p, 10)))
    posts.value = [...(twitter.data.posts || []), ...(reddit.data.posts || [])]
  } catch {
    posts.value = []
  }
}
function togglePosts(event) {
  clearInterval(postTimer)
  if (event.target.open) {
    loadPosts()
    postTimer = setInterval(loadPosts, 5000)
  }
}
onBeforeUnmount(() => clearInterval(postTimer))
</script>

<style scoped>
.head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}

.title {
  font-size: 18px;
  font-weight: 700;
  line-height: 1.5;
}

.status {
  font-size: 12px;
  border: 1px solid #000;
  padding: 2px 8px;
  white-space: nowrap;
}

.status.failed,
.status.interrupted {
  border-color: #b00020;
  color: #b00020;
}

.bar {
  height: 6px;
  background: #eee;
  margin: 20px 0;
}

.fill {
  height: 100%;
  background: #000;
  transition: width 0.4s ease;
}

.fill.failed {
  background: #b00020;
}

.problem {
  border: 1px solid #b00020;
  padding: 16px;
  margin-bottom: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-size: 14px;
}

.reason {
  font-family: inherit;
  color: #b00020;
  word-break: break-word;
}

.hint {
  color: #555;
  font-size: 13px;
}

.action-error {
  color: #b00020;
  font-size: 14px;
  margin-bottom: 16px;
}

.primary {
  align-self: flex-start;
  background: #000;
  color: #fff;
  border: none;
  padding: 8px 20px;
  font-weight: 700;
  cursor: pointer;
}

.primary:disabled {
  background: #999;
}

.stages {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 10px;
  font-size: 14px;
}

.stages li {
  display: grid;
  grid-template-columns: 24px 140px 1fr;
  align-items: baseline;
  color: #999;
}

.stages li.done,
.stages li.running {
  color: #000;
}

.stages li.failed {
  color: #b00020;
}

.stages li.running .name {
  font-weight: 700;
}

.detail {
  font-size: 13px;
  color: #555;
}

.peek {
  margin-top: 28px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.peek summary {
  cursor: pointer;
  font-size: 14px;
}

.graph-box {
  height: 480px;
  border: 1px solid #eaeaea;
  margin-top: 8px;
  position: relative;
}

.personas,
.posts {
  list-style: none;
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-size: 13px;
}

.personas li {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.personas span,
.posts .empty {
  color: #555;
}

.posts li {
  border-left: 2px solid #000;
  padding-left: 8px;
}
</style>
