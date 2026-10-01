import axios from 'axios'
import i18n from '../i18n'

const GET_RETRIES = 6

// 创建axios实例
const service = axios.create({
  // Straight to the backend on :5001 of the same host (the Docker image
  // publishes that port too). #49 had sent dev requests through the Vite
  // /api proxy after a direct POST died with ERR_CONNECTION_RESET. Measured
  // on 2026-10-01 (Vite 7, Node 24, Werkzeug 3.1, Windows) it was the other
  // way round: through the proxy about half of the responses were lost and
  // the browser re-sent one POST /api/runs three times; direct, a whole run
  // had no failed request and created one run (#66). The Werkzeug dev server
  // answered "Connection: close" there, so the browser did not reuse a
  // closed connection.
  // An https page cannot call the plain-http :5001, so it stays same-origin
  // (a reverse proxy in front). VITE_API_BASE_URL overrides both; '' goes
  // through the dev proxy.
  baseURL: import.meta.env.VITE_API_BASE_URL
    ?? (window.location.protocol === 'https:' ? '' : `http://${window.location.hostname}:5001`),
  timeout: 300000, // 5分钟超时（本体生成可能需要较长时间）
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
service.interceptors.request.use(
  config => {
    config.headers['Accept-Language'] = i18n.global.locale.value
    return config
  },
  error => {
    console.error('Request error:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器（容错重试机制）
service.interceptors.response.use(
  response => {
    const res = response.data
    
    // 如果返回的状态码不是success，则抛出错误
    if (!res.success && res.success !== undefined) {
      console.error('API Error:', res.error || res.message || 'Unknown error')
      return Promise.reject(new Error(res.error || res.message || 'Error'))
    }
    
    return res
  },
  error => {
    // A read that never got an answer (a connection reset, a backend
    // restart, #66): try again a few times. Writes are not retried; the
    // server may have acted on them.
    const config = error.config
    // No answer at all; not a timeout (that already waited minutes) and not
    // an error the server chose to send.
    const unanswered = !error.response && error.code !== 'ECONNABORTED'
    if (config && unanswered && (config.method || 'get') === 'get' && (config.retries ?? 0) < GET_RETRIES) {
      config.retries = (config.retries ?? 0) + 1
      return new Promise((resolve) => setTimeout(resolve, 250 * config.retries)).then(() => service(config))
    }
    console.error('Response error:', error)
    const apiError = error.response?.data?.error || error.response?.data?.message
    
    // 处理超时
    if (error.code === 'ECONNABORTED' && error.message.includes('timeout')) {
      console.error('Request timeout')
    }
    
    // 处理网络错误
    if (error.message === 'Network Error') {
      console.error('Network error - please check your connection')
    }

    // Axios rejects non-2xx responses before the success interceptor can
    // surface the backend's safe, actionable error message.
    if (typeof apiError === 'string' && apiError) {
      error.message = apiError
    }
    
    return Promise.reject(error)
  }
)

export default service
