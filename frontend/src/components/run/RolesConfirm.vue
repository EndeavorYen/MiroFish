<template>
  <div class="roles" data-test="roles-confirm">
    <h2>{{ $t('run.roles.title') }}</h2>
    <p class="hint">{{ $t('run.roles.hint') }}</p>
    <p v-if="loadError" class="error">{{ loadError }}</p>
    <ul v-else>
      <li v-for="role in roles" :key="role.uuid" :class="{ excluded: !role.keep }">
        <input v-model="role.keep" type="checkbox" :aria-label="$t('run.roles.keep')" />
        <input v-model="role.name" class="name" type="text" :disabled="!role.keep" />
        <span class="type">{{ role.type }}</span>
        <span class="summary">{{ role.summary }}</span>
      </li>
    </ul>
    <p v-if="saveError" class="error" data-test="roles-error">{{ saveError }}</p>
    <button class="primary" :disabled="busy || saving" data-test="confirm-roles" @click="confirm">
      {{ $t('run.roles.confirm', { count: roles.filter((r) => r.keep).length }) }}
    </button>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { editRoles, getRoles } from '../../api/runs'

const props = defineProps({ graphId: { type: String, default: '' }, busy: Boolean })
const emit = defineEmits(['confirm'])

const roles = ref([])
const original = new Map()
const loadError = ref('')
const saveError = ref('')
const saving = ref(false)

onMounted(async () => {
  try {
    const data = (await getRoles(props.graphId)).data
    roles.value = data.roles.map((role) => ({ ...role, keep: !role.excluded }))
    for (const role of roles.value) original.set(role.uuid, { name: role.name, keep: role.keep })
  } catch (error) {
    loadError.value = error.message
  }
})

function changes() {
  const ops = []
  for (const role of roles.value) {
    const before = original.get(role.uuid)
    const name = role.name.trim()
    if (role.keep && name && name !== before.name) ops.push({ op: 'rename', uuid: role.uuid, name })
    if (before.keep && !role.keep) ops.push({ op: 'exclude', uuid: role.uuid })
    if (!before.keep && role.keep) ops.push({ op: 'include', uuid: role.uuid, type: role.type })
  }
  return ops
}

async function confirm() {
  saveError.value = ''
  const ops = changes()
  if (ops.length) {
    saving.value = true
    try {
      await editRoles(props.graphId, ops)
    } catch (error) {
      saveError.value = error.message
      return
    } finally {
      saving.value = false
    }
  }
  emit('confirm')
}
</script>

<style scoped>
.roles {
  border: 1px solid #000;
  padding: 16px;
  margin-bottom: 24px;
}

h2 {
  font-size: 16px;
}

.hint {
  font-size: 13px;
  color: #555;
  margin: 6px 0 12px;
}

ul {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 360px;
  overflow-y: auto;
  margin-bottom: 12px;
}

li {
  display: grid;
  grid-template-columns: 20px 180px 140px 1fr;
  gap: 8px;
  align-items: center;
  font-size: 13px;
}

li.excluded {
  color: #999;
}

.name {
  font: inherit;
  border: 1px solid #ccc;
  padding: 4px 6px;
}

.type {
  color: #555;
}

.summary {
  color: #555;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.error {
  color: #b00020;
  font-size: 13px;
  margin-bottom: 8px;
}

.primary {
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
</style>
