<template>
  <section class="history" data-test="history">
    <div class="head">
      <h2>{{ $t('run.history.title') }}</h2>
      <input v-model="query" type="search" :placeholder="$t('run.history.search')" data-test="history-search" />
      <label class="check">
        <input v-model="showAll" type="checkbox" data-test="history-all" />
        {{ $t('run.history.showUnfinished') }}
      </label>
    </div>
    <p v-if="error" class="muted">{{ error }}</p>
    <p v-else-if="!visible.length" class="muted">{{ $t('run.history.empty') }}</p>
    <table v-else>
      <tbody>
        <tr v-for="item in visible" :key="item.run_id" @click="router.push(`/runs/${item.run_id}`)">
          <td class="when">{{ when(item.created_at) }}</td>
          <td class="what">{{ item.params.simulation_requirement }}</td>
          <td class="seeds">{{ $t('run.history.seeds', { n: item.params.seeds || 1 }) }}</td>
          <td class="status" :class="item.status">{{ $t(`run.status.${item.status}`) }}</td>
        </tr>
      </tbody>
    </table>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { listRuns } from '../../api/runs'

const router = useRouter()

// created_at is UTC; shown in the viewer's time.
const when = (iso) => new Date(iso).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })
const runs = ref([])
const query = ref('')
const showAll = ref(false)
const error = ref('')

// Finished runs by default; failed, stopped and running ones on request.
const visible = computed(() => runs.value.filter((item) =>
  (showAll.value || item.status === 'completed')
  && (!query.value.trim() || (item.params.simulation_requirement || '').includes(query.value.trim()))))

onMounted(async () => {
  try {
    runs.value = (await listRuns()).data
  } catch (err) {
    error.value = err.message
  }
})
</script>

<style scoped>
.head {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

h2 {
  font-size: 15px;
  margin-right: auto;
}

input[type='search'] {
  font: inherit;
  font-size: 13px;
  border: 1px solid #ccc;
  padding: 6px 8px;
}

.check {
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.muted {
  font-size: 13px;
  color: #777;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

tr {
  cursor: pointer;
  border-top: 1px solid #eaeaea;
}

tr:hover {
  background: #f7f7f7;
}

td {
  padding: 8px 6px;
  vertical-align: top;
}

.when,
.seeds,
.status {
  white-space: nowrap;
  color: #555;
}

.what {
  width: 100%;
}

.status.failed,
.status.interrupted {
  color: #b00020;
}
</style>
