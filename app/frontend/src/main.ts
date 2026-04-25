import { createApp } from 'vue'
import './assets/style.css'

import {library} from '@fortawesome/fontawesome-svg-core'
import {FontAwesomeIcon} from '@fortawesome/vue-fontawesome'
import {fas} from '@fortawesome/free-solid-svg-icons'

import App from './App.vue'
import router from './router'

library.add(fas);
const app = createApp(App)
app.component("Icon", FontAwesomeIcon);
app.use(router)
app.mount('#app')
