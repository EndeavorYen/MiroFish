<template>
  <section class="run-result" data-test="run-result">
    <div class="head">
      <h1 class="title">{{ run.params?.simulation_requirement }}</h1>
      <span class="meta">{{ $t('run.result.meta', { seeds: seedCount, rounds: run.artifacts?.rounds ?? run.params?.max_rounds }) }}</span>
    </div>

    <p v-if="error" class="error" data-test="result-error">{{ error }}</p>
    <p v-else-if="!metrics" class="muted">{{ $t('run.loading') }}</p>

    <template v-else>
      <!-- Confidence first: each conclusion with how far it can be trusted. -->
      <p v-if="!scan" class="muted" data-test="no-scan">{{ $t('run.result.noScan', { reason: metrics.scan_error || '' }) }}</p>
      <div v-else class="cards" data-test="confidence-cards">
        <article class="card" data-test="card-main-camp">
          <header>
            <h2>{{ $t('run.result.mainCamp') }}</h2>
            <span class="badge" :class="scan.main_camp.confidence">{{ $t(`run.confidence.${scan.main_camp.confidence}`) }}</span>
          </header>
          <p class="value">{{ $t(`run.camp.${scan.main_camp.value}`) }}</p>
          <p class="sub">
            {{ $t('run.result.campCounts', { support: scan.main_camp.counts.support, neutral: scan.main_camp.counts.neutral, oppose: scan.main_camp.counts.oppose }) }}
          </p>
          <p class="evidence">{{ scan.main_camp.evidence }}</p>
        </article>

        <article class="card" data-test="card-tendency">
          <header>
            <h2>{{ $t('run.result.tendency') }}</h2>
            <span class="badge" :class="scan.tendency.confidence">{{ $t(`run.confidence.${scan.tendency.confidence}`) }}</span>
          </header>
          <p class="value">{{ scan.tendency.value.toFixed(2) }}</p>
          <div class="scale">
            <span>{{ $t('run.camp.oppose') }}</span>
            <div class="track"><div class="dot" :style="{ left: `${scan.tendency.value * 100}%` }" /></div>
            <span>{{ $t('run.camp.support') }}</span>
          </div>
          <p class="evidence">{{ scan.tendency.evidence }}</p>
        </article>

        <article class="card" data-test="card-trend">
          <header>
            <h2>{{ $t('run.result.trend') }}</h2>
            <span class="badge" :class="scan.trend.confidence">{{ $t(`run.confidence.${scan.trend.confidence}`) }}</span>
          </header>
          <p class="value" :class="{ quiet: scan.trend.value == null }">{{ trendText }}</p>
          <p v-if="scan.trend.rounds" class="sub">{{ $t('run.result.trendRounds', { from: scan.trend.rounds[0], to: scan.trend.rounds[1] }) }}</p>
          <p class="evidence">{{ scan.trend.evidence }}</p>
        </article>

        <article class="card" data-test="card-ranking">
          <header>
            <h2>{{ $t('run.result.ranking') }}</h2>
            <span class="badge" :class="scan.ranking.confidence">{{ $t(`run.confidence.${scan.ranking.confidence}`) }}</span>
          </header>
          <p v-if="scan.ranking.indistinct" class="sub">{{ $t('run.result.indistinct') }}</p>
          <template v-else>
            <p class="sub"><strong>{{ $t('run.result.mostSupportive') }}</strong> {{ names(scan.ranking.most_supportive) }}</p>
            <p class="sub"><strong>{{ $t('run.result.mostOpposed') }}</strong> {{ names(scan.ranking.most_opposed) }}</p>
          </template>
          <p class="evidence">{{ scan.ranking.evidence }}</p>
        </article>
      </div>

      <!-- How far the seeds agree (#63). -->
      <section v-if="consistency" class="block" data-test="consistency">
        <h2>{{ $t('run.result.consistencyTitle', { seeds: consistency.seeds }) }}</h2>
        <ul class="facts">
          <li>{{ $t('run.result.seedsAgree', { camp: $t(`run.camp.${consistency.main_camp.value}`), agree: consistency.main_camp.agree, seeds: consistency.seeds }) }}</li>
          <li>{{ $t('run.result.tendencySpread', { mean: consistency.tendency.mean.toFixed(2), sd: consistency.tendency.sd.toFixed(2) }) }}</li>
          <li v-if="consistency.trend.mean != null">{{ $t('run.result.trendSpread', { mean: signed(consistency.trend.mean), sd: consistency.trend.sd.toFixed(2) }) }}</li>
          <li v-if="consistency.ranking.stability != null">{{ $t('run.result.rankingStability', { value: consistency.ranking.stability.toFixed(2) }) }}</li>
        </ul>
        <p v-if="consistency.confirm.length" class="warn" data-test="consistency-confirm">
          {{ $t('run.result.confirmWith') }} {{ consistency.confirm.join('；') }}
        </p>
        <p v-else class="ok">{{ $t('run.result.seedsConsistent') }}</p>
      </section>
      <p v-else-if="run.artifacts?.consistency_error" class="muted">{{ $t('run.result.consistencyFailed', { reason: run.artifacts.consistency_error }) }}</p>

      <section v-if="topPosts.length" class="block" data-test="top-posts">
        <h2>{{ $t('run.result.topPosts') }}</h2>
        <ul class="posts">
          <li v-for="(post, i) in topPosts" :key="i">
            <p class="post-meta">{{ post.author }} · {{ $t(`run.platform.${post.platform}`) }} · {{ $t('run.result.postStats', { shares: post.reposts_and_quotes, likes: post.likes }) }}</p>
            <p>{{ post.content }}</p>
          </li>
        </ul>
      </section>

      <details v-if="reportText" class="block report">
        <summary>{{ $t('run.result.fullReport') }}</summary>
        <div class="report-text">{{ reportText }}</div>
      </details>
    </template>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { getReport, getReportMetrics } from '../../api/runs'

const props = defineProps({ run: { type: Object, required: true } })
const { t } = useI18n()

const metrics = ref(null)
const reportText = ref('')
const error = ref('')

const scan = computed(() => metrics.value?.scan || null)
const consistency = computed(() => props.run.artifacts?.consistency || null)
const seedCount = computed(() => props.run.artifacts?.seed_simulations?.length || props.run.params?.seeds || 1)

const signed = (value) => `${value > 0 ? '+' : ''}${value.toFixed(2)}`
const names = (list) => (list || []).map((r) => r.name).join('、')
const trendText = computed(() => {
  const value = scan.value?.trend?.value
  if (value == null) return t('run.result.trendNone')
  if (Math.abs(value) < 0.02) return `${t('run.result.trendFlat')} (${signed(value)})`
  return `${value > 0 ? t('run.result.trendUp') : t('run.result.trendDown')} (${signed(value)})`
})
const topPosts = computed(() => {
  const spread = metrics.value?.spread || {}
  return Object.entries(spread).flatMap(([platform, posts]) =>
    (posts || []).slice(0, 3).map((post) => ({ ...post, platform })))
})

onMounted(async () => {
  const reportId = props.run.artifacts?.report_id
  if (!reportId) {
    error.value = t('run.result.noReport')
    return
  }
  // The full text is optional and loads on its own: the cards do not wait for it.
  getReport(reportId).then((r) => { reportText.value = r.data?.markdown_content || '' }).catch(() => {})
  try {
    metrics.value = (await getReportMetrics(reportId)).data
  } catch (err) {
    error.value = err.message
  }
})
</script>

<style scoped>
.head {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 24px;
}

.title {
  font-size: 18px;
  font-weight: 700;
  line-height: 1.5;
}

.meta,
.muted {
  font-size: 13px;
  color: #555;
}

.error {
  color: #b00020;
}

.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 12px;
}

.card {
  border: 1px solid #000;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.card header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card h2 {
  font-size: 13px;
  font-weight: 600;
}

.badge {
  font-size: 11px;
  padding: 1px 6px;
  border: 1px solid currentColor;
}

.badge.high {
  color: #0a7a2f;
}

.badge.medium {
  color: #8a6d00;
}

.badge.low,
.badge.none {
  color: #b00020;
}

.badge.reference {
  color: #555;
}

.value {
  font-size: 22px;
  font-weight: 700;
}

.value.quiet {
  font-size: 14px;
  font-weight: 400;
  color: #555;
}

.sub {
  font-size: 13px;
}

.evidence {
  font-size: 11px;
  color: #777;
  margin-top: auto;
}

.scale {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 6px;
  align-items: center;
  font-size: 11px;
  color: #555;
}

.track {
  height: 4px;
  background: #ddd;
  position: relative;
}

.dot {
  position: absolute;
  top: -4px;
  width: 12px;
  height: 12px;
  margin-left: -6px;
  background: #000;
  border-radius: 50%;
}

.block {
  margin-top: 28px;
}

.block h2,
.report summary {
  font-size: 15px;
  font-weight: 700;
  margin-bottom: 10px;
}

.facts {
  font-size: 14px;
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.warn {
  margin-top: 10px;
  font-size: 13px;
  color: #8a6d00;
}

.ok {
  margin-top: 10px;
  font-size: 13px;
  color: #0a7a2f;
}

.posts {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 12px;
  font-size: 14px;
  line-height: 1.6;
}

.posts li {
  border-left: 2px solid #000;
  padding-left: 10px;
}

.post-meta {
  font-size: 12px;
  color: #555;
}

.report summary {
  cursor: pointer;
}

.report-text {
  white-space: pre-wrap;
  font-size: 13px;
  line-height: 1.7;
  border: 1px solid #eaeaea;
  padding: 16px;
}
</style>
