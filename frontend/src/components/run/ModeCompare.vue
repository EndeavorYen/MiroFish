<template>
  <section class="compare" data-test="mode-compare">
    <h2>{{ $t('run.compare.title', { mode: modeLabel(run.params.profile) }) }}</h2>
    <p v-if="error" class="muted">{{ error }}</p>
    <p v-else-if="!theirs" class="muted">{{ $t('run.loading') }}</p>
    <template v-else>
      <table>
        <thead>
          <tr>
            <th />
            <th>
              <router-link :to="`/runs/${run.params.confirms}`">{{ modeLabel(original.params.profile) }}</router-link>
            </th>
            <th>{{ modeLabel(run.params.profile) }}</th>
            <th />
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in rows" :key="row.key" :data-row="row.key">
            <th>{{ $t(`run.result.${row.key}`) }}</th>
            <td>{{ row.theirs }}</td>
            <td>{{ row.ours }}</td>
            <td :class="row.agree == null ? 'na' : row.agree ? 'agree' : 'differ'">
              {{ row.agree == null ? $t('run.compare.na') : row.agree ? $t('run.compare.agree') : $t('run.compare.differ') }}
            </td>
          </tr>
        </tbody>
      </table>
      <p class="muted">{{ $t(acrossSeeds ? 'run.compare.acrossSeeds' : 'run.compare.firstSeed') }} {{ $t('run.compare.note') }}</p>
    </template>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { MODES, getReportMetrics, getRun } from '../../api/runs'

// This run confirms another one (``params.confirms``) in another mode: the
// conclusions side by side, and whether they agree.
const props = defineProps({ run: { type: Object, required: true }, scan: { type: Object, default: null } })

// Both runs over several seeds: compare the seed means (seed noise averages
// out); otherwise the report's own conclusions, from the first seed.
const theirSeeds = ref(null)
const acrossSeeds = computed(() => Boolean(theirSeeds.value && props.run.artifacts?.consistency))
const fromSeeds = (c) => ({
  main_camp: { value: c.main_camp.value },
  tendency: { value: c.tendency.mean },
  trend: { value: c.trend.mean },
  ranking: { indistinct: c.ranking.indistinct, most_opposed: (c.ranking.most_opposed || []).map((name) => ({ name })) },
})
const { t } = useI18n()

const original = ref(null)
const theirs = ref(null)
const error = ref('')

const TENDENCY_CLOSE = 0.1
const modeLabel = (mode) => t(MODES.includes(mode) ? `run.mode.${mode}` : 'run.mode.unknown')
const camp = (s) => (s?.main_camp ? t(`run.camp.${s.main_camp.value}`) : '—')
const number = (v, signed = false) => (v == null ? '—' : `${signed && v > 0 ? '+' : ''}${v.toFixed(2)}`)
const firstOpposed = (s) => s?.ranking?.most_opposed?.[0]?.name || '—'

const rows = computed(() => {
  const a = acrossSeeds.value ? fromSeeds(theirSeeds.value) : theirs.value
  const b = acrossSeeds.value ? fromSeeds(props.run.artifacts.consistency) : props.scan
  if (!a || !b) return []
  const trendKnown = a.trend?.value != null && b.trend?.value != null
  return [
    { key: 'mainCamp', theirs: camp(a), ours: camp(b), agree: a.main_camp.value === b.main_camp.value },
    {
      key: 'tendency', theirs: number(a.tendency.value), ours: number(b.tendency.value),
      agree: Math.abs(a.tendency.value - b.tendency.value) <= TENDENCY_CLOSE,
    },
    {
      key: 'trend', theirs: number(a.trend?.value, true), ours: number(b.trend?.value, true),
      agree: trendKnown ? Math.sign(a.trend.value) === Math.sign(b.trend.value) : null,
    },
    {
      key: 'mostOpposedFirst', theirs: firstOpposed(a), ours: firstOpposed(b),
      agree: a.ranking?.indistinct || b.ranking?.indistinct ? null : firstOpposed(a) === firstOpposed(b),
    },
  ]
})

onMounted(async () => {
  try {
    original.value = (await getRun(props.run.params.confirms)).data
    theirSeeds.value = original.value.artifacts?.consistency || null
    const reportId = original.value.artifacts?.report_id
    if (!reportId) throw new Error(t('run.result.noReport'))
    theirs.value = (await getReportMetrics(reportId)).data.scan
    if (!theirs.value) throw new Error(t('run.result.noScanReason'))
  } catch (err) {
    error.value = t('run.compare.unavailable', { reason: err.message })
  }
})
</script>

<style scoped>
.compare {
  margin-top: 28px;
}

h2 {
  font-size: 15px;
  font-weight: 700;
  margin-bottom: 10px;
}

table {
  border-collapse: collapse;
  font-size: 14px;
  width: 100%;
}

th,
td {
  text-align: left;
  padding: 6px 8px;
  border-bottom: 1px solid #eaeaea;
}

thead th {
  font-size: 12px;
  color: #555;
}

thead a {
  color: inherit;
}

.agree {
  color: #0a7a2f;
}

.differ {
  color: #b00020;
}

.na,
.muted {
  color: #777;
}

.muted {
  font-size: 12px;
  margin-top: 8px;
}
</style>
