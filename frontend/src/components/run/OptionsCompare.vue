<template>
  <section class="options" data-test="options-compare">
    <h2>{{ $t('run.options.title') }}</h2>
    <table>
      <thead>
        <tr>
          <th>{{ $t('run.options.option') }}</th>
          <th>{{ $t('run.result.tendency') }}</th>
          <th>{{ $t('run.options.opposeShare') }}</th>
          <th>{{ $t('run.options.vsTendency') }}</th>
          <th>{{ $t('run.options.vsOppose') }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows" :key="row.option" :data-option="row.option">
          <th>
            {{ row.option }}
            <span v-if="!row.vs_baseline" class="tag">{{ $t('run.options.baseline') }}</span>
          </th>
          <td>{{ row.tendency.toFixed(2) }}</td>
          <td>{{ percent(row.oppose_share) }}</td>
          <td :class="cls(row.vs_baseline?.tendency)">{{ diff(row.vs_baseline?.tendency) }}</td>
          <td :class="cls(row.vs_baseline?.oppose_share)">{{ diff(row.vs_baseline?.oppose_share, 100, $t('run.options.points')) }}</td>
        </tr>
      </tbody>
    </table>
    <p class="muted">{{ $t('run.options.note') }}</p>

    <details class="posts">
      <summary>{{ $t('run.options.mostOpposed') }}</summary>
      <div v-for="row in rows" :key="row.option" class="option-posts">
        <strong>{{ row.option }}</strong>
        <ul>
          <li v-for="(post, i) in row.most_opposed_posts" :key="i">{{ post.text }} <span class="muted">({{ post.stance.toFixed(2) }})</span></li>
        </ul>
      </div>
    </details>
  </section>
</template>

<script setup>
import { useI18n } from 'vue-i18n'

// scripts/compare_options.py's table: each option against the baseline, the
// difference paired seed by seed (#56, #66).
defineProps({ rows: { type: Array, required: true } })
const { t } = useI18n()

const percent = (v) => `${Math.round(v * 100)}%`
const cls = (block) => (!block || block.mean == null ? '' : block.distinct ? 'distinct' : 'unclear')
function diff(block, scale = 1, unit = '') {
  if (!block || block.mean == null) return '—'
  const digits = scale === 1 ? 2 : 1
  const value = block.mean * scale
  const text = `${value > 0 ? '+' : ''}${value.toFixed(digits)}${unit}`
  const seeds = t('run.options.sameSign', { same: block.same_sign, seeds: block.seeds })
  // Fewer than 3 seeds never tell options apart (backend option_compare).
  const verdict = block.distinct ? '' : `, ${t(block.seeds < 3 ? 'run.options.fewSeeds' : 'run.options.unclear')}`
  return `${text} (${seeds}${verdict})`
}
</script>

<style scoped>
.options {
  margin-top: 28px;
}

h2 {
  font-size: 15px;
  font-weight: 700;
  margin-bottom: 10px;
}

table {
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
}

th,
td {
  text-align: left;
  padding: 6px 8px;
  border-bottom: 1px solid #eaeaea;
  vertical-align: top;
}

thead th {
  font-size: 12px;
  color: #555;
  font-weight: 600;
}

.tag {
  font-size: 11px;
  border: 1px solid #999;
  padding: 0 4px;
  margin-left: 4px;
  color: #555;
}

.distinct {
  font-weight: 700;
}

.unclear,
.muted {
  color: #777;
}

.muted {
  font-size: 12px;
  margin-top: 8px;
}

.posts {
  margin-top: 12px;
  font-size: 13px;
}

.posts summary {
  cursor: pointer;
}

.option-posts {
  margin-top: 8px;
}

.option-posts ul {
  padding-left: 18px;
  margin-top: 4px;
}
</style>
