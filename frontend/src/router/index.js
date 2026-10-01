import { createRouter, createWebHistory } from 'vue-router'
import RunPage from '../views/RunPage.vue'

// One page in three states (#66): input, a run in progress, its result.
const routes = [
  { path: '/', name: 'Home', component: RunPage },
  { path: '/runs/:runId', name: 'Run', component: RunPage, props: true },
  { path: '/:pathMatch(.*)*', redirect: '/' },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
