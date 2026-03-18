<template>
  <div class="language-switcher">
    <label class="language-label" for="locale-select">{{ t('language.label') }}</label>
    <select
      id="locale-select"
      class="language-select"
      :value="locale"
      @change="onChangeLocale"
    >
      <option v-for="item in localeOptions" :key="item.value" :value="item.value">
        {{ item.label }}
      </option>
    </select>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { getSupportedLocales, setLocale } from '../i18n'

const { t, locale } = useI18n()

const localeOptions = computed(() =>
  getSupportedLocales().map((value) => ({
    value,
    label: value === 'zh-TW' ? t('language.zhTW') : t('language.zhCN')
  }))
)

const onChangeLocale = (event) => {
  setLocale(event.target.value)
}
</script>

<style scoped>
.language-switcher {
  position: fixed;
  top: 10px;
  right: 14px;
  z-index: 9999;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border: 1px solid #e5e5e5;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(4px);
}

.language-label {
  font-size: 12px;
  color: #555;
}

.language-select {
  border: 1px solid #d8d8d8;
  border-radius: 6px;
  background: #fff;
  padding: 4px 8px;
  font-size: 12px;
  color: #222;
  cursor: pointer;
}
</style>
