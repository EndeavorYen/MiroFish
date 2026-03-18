import { createI18n } from 'vue-i18n'
import zhTW from './locales/zh-TW.json'
import zhCN from './locales/zh-CN.json'

const LOCALE_STORAGE_KEY = 'mirofish_locale'
const SUPPORTED_LOCALES = ['zh-TW', 'zh-CN']
const DEFAULT_LOCALE = 'zh-TW'

const messages = {
  'zh-TW': zhTW,
  'zh-CN': zhCN
}

function normalizeLocale(input) {
  if (!input) {
    return null
  }
  const value = String(input).toLowerCase()
  if (value === 'zh-tw' || value.startsWith('zh-hant')) {
    return 'zh-TW'
  }
  if (value === 'zh-cn' || value.startsWith('zh-hans') || value.startsWith('zh')) {
    return 'zh-CN'
  }
  return null
}

function resolveInitialLocale() {
  const stored = normalizeLocale(window.localStorage.getItem(LOCALE_STORAGE_KEY))
  if (stored) {
    return stored
  }
  return DEFAULT_LOCALE
}

export const i18n = createI18n({
  legacy: false,
  globalInjection: true,
  locale: resolveInitialLocale(),
  fallbackLocale: DEFAULT_LOCALE,
  messages
})

export function setLocale(locale) {
  if (!SUPPORTED_LOCALES.includes(locale)) {
    return
  }
  i18n.global.locale.value = locale
  window.localStorage.setItem(LOCALE_STORAGE_KEY, locale)
  document.documentElement.lang = locale
}

export function getLocale() {
  return i18n.global.locale.value
}

export function getSupportedLocales() {
  return SUPPORTED_LOCALES.slice()
}

document.documentElement.lang = getLocale()

export default i18n
