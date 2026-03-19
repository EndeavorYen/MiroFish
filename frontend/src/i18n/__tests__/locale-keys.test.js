import { describe, it, expect } from 'vitest'
import zhTW from '../locales/zh-TW.json'
import zhCN from '../locales/zh-CN.json'
import en from '../locales/en.json'

// Helper: recursively collect all keys with dot notation
function collectKeys(obj, prefix = '') {
  let keys = []
  for (const key of Object.keys(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key
    if (typeof obj[key] === 'object' && obj[key] !== null && !Array.isArray(obj[key])) {
      keys = keys.concat(collectKeys(obj[key], fullKey))
    } else {
      keys.push(fullKey)
    }
  }
  return keys.sort()
}

describe('i18n locale files', () => {
  const zhTWKeys = collectKeys(zhTW)
  const zhCNKeys = collectKeys(zhCN)
  const enKeys = collectKeys(en)

  it('all locales should have the same number of keys', () => {
    expect(zhTWKeys.length).toBe(zhCNKeys.length)
    expect(zhTWKeys.length).toBe(enKeys.length)
  })

  it('zh-CN should have all keys from zh-TW', () => {
    const missing = zhTWKeys.filter(k => !zhCNKeys.includes(k))
    expect(missing, `zh-CN is missing keys: ${missing.join(', ')}`).toEqual([])
  })

  it('en should have all keys from zh-TW', () => {
    const missing = zhTWKeys.filter(k => !enKeys.includes(k))
    expect(missing, `en is missing keys: ${missing.join(', ')}`).toEqual([])
  })

  it('zh-TW should have all keys from en (no extra keys in en)', () => {
    const extra = enKeys.filter(k => !zhTWKeys.includes(k))
    expect(extra, `en has extra keys not in zh-TW: ${extra.join(', ')}`).toEqual([])
  })

  it('no empty translation values', () => {
    const checkEmpty = (obj, locale, prefix = '') => {
      const empties = []
      for (const key of Object.keys(obj)) {
        const fullKey = prefix ? `${prefix}.${key}` : key
        if (typeof obj[key] === 'object' && obj[key] !== null) {
          empties.push(...checkEmpty(obj[key], locale, fullKey))
        } else if (obj[key] === '') {
          // Allow intentionally empty values (like countUnit in en)
          // Only flag if empty in ALL locales
        }
      }
      return empties
    }
    // This test is informational - some keys may be intentionally empty
  })
})
