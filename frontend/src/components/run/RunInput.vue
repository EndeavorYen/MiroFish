<template>
  <section class="run-input">
    <h1 class="title">{{ $t('run.input.title') }}</h1>
    <p class="lead">{{ $t('run.input.lead') }}</p>

    <form @submit.prevent="start">
      <div class="source-tabs" role="tablist">
        <button type="button" role="tab" :class="{ active: source === 'text' }" @click="source = 'text'">
          {{ $t('run.input.pasteText') }}
        </button>
        <button type="button" role="tab" :class="{ active: source === 'file' }" @click="source = 'file'">
          {{ $t('run.input.uploadFile') }}
        </button>
      </div>

      <textarea
        v-if="source === 'text'"
        v-model="text"
        class="document"
        data-test="document-text"
        :placeholder="$t('run.input.documentPlaceholder')"
        rows="8"
      />
      <label v-else class="file-drop">
        <input type="file" accept=".txt,.md,.markdown,.pdf" data-test="document-file" @change="pickFile" />
        <span>{{ file ? file.name : $t('run.input.fileHint') }}</span>
      </label>

      <label class="field-label" for="requirement">{{ $t('run.input.requirement') }}</label>
      <textarea
        id="requirement"
        v-model="requirement"
        class="requirement"
        data-test="requirement"
        :placeholder="$t('run.input.requirementPlaceholder')"
        rows="3"
      />

      <details class="advanced">
        <summary>{{ $t('run.input.advanced') }}</summary>
        <div class="advanced-grid">
          <label>
            {{ $t('run.input.rounds') }}
            <input v-model.number="maxRounds" type="number" min="1" max="200" data-test="rounds" />
          </label>
          <label>
            {{ $t('run.input.seeds') }}
            <input v-model.number="seeds" type="number" min="1" max="8" data-test="seeds" />
          </label>
          <label class="check">
            <input v-model="confirmRoles" type="checkbox" data-test="confirm-roles" />
            {{ $t('run.input.confirmRoles') }}
          </label>
        </div>
        <p class="hint">{{ $t('run.input.seedsHint') }}</p>
      </details>

      <p v-if="error" class="error" data-test="input-error">{{ error }}</p>
      <button class="start" type="submit" :disabled="!ready || starting" data-test="start">
        {{ starting ? $t('run.input.starting') : $t('run.input.start') }}
      </button>
    </form>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue'
import { createRun } from '../../api/runs'

const emit = defineEmits(['started'])

const source = ref('text')
const text = ref('')
const file = ref(null)
const requirement = ref('')
const maxRounds = ref(24)
const seeds = ref(3)
const confirmRoles = ref(false)
const starting = ref(false)
const error = ref('')

const ready = computed(() =>
  requirement.value.trim() && (source.value === 'text' ? text.value.trim() : file.value))

function pickFile(event) {
  file.value = event.target.files?.[0] || null
}

async function start() {
  starting.value = true
  error.value = ''
  try {
    const response = await createRun({
      file: source.value === 'file' ? file.value : null,
      text: text.value,
      requirement: requirement.value.trim(),
      maxRounds: maxRounds.value,
      seeds: seeds.value,
      confirmRoles: confirmRoles.value,
    })
    emit('started', response.data.run_id)
  } catch (err) {
    error.value = err.message
  } finally {
    starting.value = false
  }
}
</script>

<style scoped>
.title {
  font-size: 22px;
  font-weight: 700;
}

.lead {
  margin: 8px 0 20px;
  color: #555;
  font-size: 14px;
  line-height: 1.6;
}

form {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.source-tabs {
  display: flex;
  gap: 4px;
}

.source-tabs button {
  border: 1px solid #000;
  background: #fff;
  padding: 6px 12px;
  font-size: 13px;
  cursor: pointer;
}

.source-tabs button.active {
  background: #000;
  color: #fff;
}

textarea,
input[type='number'] {
  font: inherit;
  font-size: 14px;
  border: 1px solid #ccc;
  padding: 10px;
  width: 100%;
}

textarea:focus,
input:focus {
  outline: 2px solid #000;
}

.file-drop {
  border: 1px dashed #999;
  padding: 24px;
  font-size: 14px;
  color: #555;
  display: flex;
  flex-direction: column;
  gap: 8px;
  cursor: pointer;
}

.field-label {
  font-size: 13px;
  font-weight: 600;
  margin-top: 8px;
}

.advanced summary {
  cursor: pointer;
  font-size: 13px;
}

.advanced-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 12px;
}

.advanced-grid label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 13px;
  width: 140px;
}

.advanced-grid label.check {
  flex-direction: row;
  align-items: center;
  width: auto;
}

.hint {
  font-size: 12px;
  color: #777;
  margin-top: 8px;
}

.error {
  color: #b00020;
  font-size: 14px;
}

.start {
  align-self: flex-start;
  background: #000;
  color: #fff;
  border: none;
  padding: 12px 32px;
  font-size: 15px;
  font-weight: 700;
  cursor: pointer;
}

.start:disabled {
  background: #999;
  cursor: not-allowed;
}
</style>
