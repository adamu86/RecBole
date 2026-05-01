<script setup lang="ts">
import {
  fetchTracks,
  fetchRecommendations,
  fetchModels,
  setModel,
  fetchStatus,
  fetchMetrics,
  fetchLog,
  type TrackItem,
  type TrackRecommendation,
  type ParsedLogResponse,
  type EpochData,
} from "./service";
import { ref, computed, onMounted, watch, nextTick } from "vue";
import { Chart, LineController, LineElement, PointElement, LinearScale, CategoryScale, Legend, Tooltip, Filler } from 'chart.js';

Chart.register(LineController, LineElement, PointElement, LinearScale, CategoryScale, Legend, Tooltip, Filler);

const page = ref<number>(1);
const limit = ref<number>(25);
const offset = computed(() => (page.value - 1) * limit.value);
const searchQuery = ref<string>("Avicii");
const k = ref<number>(30);
const interval = ref<number>(2);
const autoContinue = ref<boolean>(false);
const autoPlaying = ref<boolean>(false);
const selectedTracks = ref<TrackItem[]>([]);
const tracks = ref<TrackItem[]>([]);
const recommendations = ref<TrackRecommendation[]>([]);
const isFetchingTracks = ref<boolean>(false);

const models = ref<string[]>([]);
const currentModel = ref<string>("");
const loadingModel = ref<boolean>(false);

const currentMetricsTable = ref<{ results_1: {}, results_N: {} }>({results_1: {}, results_N: {}});
const showMetricsTable = ref<boolean>(false);
    
const currentLog = ref<ParsedLogResponse>({ epochs: [] });
const showLog = ref<boolean>(false);
const chartCanvas = ref<HTMLCanvasElement | null>(null);
const selectedMetrics = ref<string[]>(['recall@10', 'precision@10', 'mrr@10', 'map@10', 'hit@10', 'ndcg@10']);
let chartInstance: Chart | null = null;

const CHART_COLORS = [
  '#6366f1', '#f43f5e', '#10b981', '#f59e0b', '#3b82f6',
  '#8b5cf6', '#ec4899', '#14b8a6', '#ef4444', '#84cc16',
  '#06b6d4', '#d946ef', '#f97316', '#22d3ee', '#a3e635',
];

const allMetricKeys = computed(() => {
  const epochs = currentLog.value.epochs;
  if (!epochs.length) return [];
  const keys = new Set<string>();
  keys.add('train_loss');
  keys.add('valid_score');
  epochs.forEach(e => Object.keys(e.metrics).forEach(k => keys.add(k)));
  return Array.from(keys);
});

const toggleMetric = (m: string) => {
  const i = selectedMetrics.value.indexOf(m);
  if (i >= 0) selectedMetrics.value.splice(i, 1);
  else selectedMetrics.value.push(m);
  buildChart();
};

const buildChart = () => {
  if (!chartCanvas.value) return;
  const epochs = currentLog.value.epochs;
  if (!epochs.length) return;

  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  const labels = epochs.map(e => `Epoch ${e.epoch + 1}`);
  const active = selectedMetrics.value.filter(m => allMetricKeys.value.includes(m));

  const datasets = active.map((metric) => {
    const colorIdx = allMetricKeys.value.indexOf(metric);
    const color = CHART_COLORS[colorIdx % CHART_COLORS.length];
    const data = epochs.map(e => {
      if (metric === 'train_loss') return e.train_loss ?? NaN;
      if (metric === 'valid_score') return e.valid_score ?? NaN;
      return e.metrics[metric] ?? NaN;
    });
    const isLoss = metric === 'train_loss';
    return {
      label: metric,
      data,
      borderColor: color,
      backgroundColor: color + '22',
      pointBackgroundColor: color,
      pointRadius: 1,
      pointHoverRadius: 4,
      borderWidth: 2,
      tension: 0.3,
      yAxisID: isLoss ? 'yLoss' : 'y',
    };
  });

  const hasLoss = active.includes('train_loss');

  chartInstance = new Chart(chartCanvas.value, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      layout: {
        padding: 10,
      },
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { boxWidth: 12, padding: 20, font: { size: 15 }, usePointStyle: true, pointStyle: 'circle', boxHeight: 10 },
        },
        tooltip: {
          backgroundColor: '#1e293b',
          titleFont: { size: 15 },
          bodyFont: { size: 15 },
          padding: 5,
        },
      },
      scales: {
        x: {
          grid: { color: '#f1f5f9' },
          ticks: {
            font: { size: 12 },
            maxRotation: 45,
            autoSkip: false,
            callback: function (value, index) {
              return (index + 1) % 5 === 0 || index === 0 
              ? this.getLabelForValue(value) 
              : '';
            }
          },
        },
        y: {
          position: 'left',
          beginAtZero: true,
          grid: { color: '#f1f5f9' },
          ticks: { font: { size: 12 }, stepSize: 0.005 },
          title: { display: true, text: 'Metrics', font: { size: 15 } },
        },
        ...(hasLoss ? {
          yLoss: {
            position: 'right' as const,
            grid: { drawOnChartArea: false },
            ticks: { font: { size: 9 } },
            title: { display: true, text: 'Train Loss', font: { size: 15 } },
          },
        } : {}),
      },
    },
  });
};

const toggleChart = async () => {
  showLog.value = !showLog.value;
  if (showLog.value) {
    await nextTick();
    buildChart();
  }
};

const getTracks = async () => {
  isFetchingTracks.value = true;
  try {
    const response = await fetchTracks(
      offset.value,
      limit.value,
      searchQuery.value,
    );
    tracks.value = response.tracks;
  } finally {
    isFetchingTracks.value = false;
  }
};

const getRecommendations = async () => {
  const response = await fetchRecommendations(
    selectedTracks.value.map((t: TrackItem) => parseInt(t.id)),
    k.value,
  );
  recommendations.value = response.recommendations;
};

const autoPlay = async () => {
  if (!autoPlaying.value) {
    return;
  }

  await getRecommendations();
  const selectedIds = new Set(selectedTracks.value.map((t) => t.id));
  const next = recommendations.value.find(
    (r) => !selectedIds.has(r.track_id.toString()),
  );
  if (next) {
    setTimeout(() => {
      selectedTracks.value.push({
        id: next.track_id.toString(),
        name: next.name,
      });
      autoPlay();
    }, interval.value * 1000);
  }
};

const nextPage = () => {
  page.value++;
  getTracks();
};

const prevPage = () => {
  if (page.value > 1) {
    page.value--;
    getTracks();
  }
};

const resetPage = () => {
  page.value = 1;
};

const selectTrack = (trackId: string, trackName: string) => {
  selectedTracks.value.push({
    id: trackId,
    name: trackName,
  })
}

const grayedOut = (trackName: string) => selectedTracks.value.some((t) => t.name === trackName) ? 'opacity-25' : '';

const getModels = async () => {
  models.value = await fetchModels();
  models.value.sort();
};

const setNewModel = async (modelPath: string) => {
  if (modelPath) {
    loadingModel.value = true;
    try {
      resetPage();
      showMetricsTable.value = false;
      showLog.value = false;
      tracks.value = [];
      selectedTracks.value = [];
      recommendations.value = [];
      autoContinue.value = false;
      autoPlaying.value = false;
      currentModel.value = modelPath;
      await setModel(modelPath);
      await init();
    } finally {
      loadingModel.value = false;
    }
  }
};

const getMetrics = async () => {
  currentMetricsTable.value = await fetchMetrics();
};

const getLog = async () => {
  currentLog.value = await fetchLog();
  console.log(currentLog.value);
};

const init = async () => {
  const status = await fetchStatus();
  currentModel.value = status.model;
  for (const [key, value] of Object.entries(status)) {
    console.log(`${key}: ${value}`);
  }
  getModels();
  getTracks();
  getMetrics();
  getLog();
}

watch(() => selectedTracks.value.length, () => {
  if (autoContinue.value && !autoPlaying.value) {
    getRecommendations();
  }
});

onMounted(init);
</script>

<template>
  <main class="grid grid-rows-[auto_1fr_auto] grid-cols-3 gap-2 p-4 h-screen overflow-hidden" :class="loadingModel ? 'pointer-events-none opacity-50' : ''">
    <h2 class="col-span-3 mb-2 flex flex-row justify-between">
      <span class="text-3xl font-bold">
        Music Recommender
      </span>
      <div v-if="models.length > 0" class="flex items-center">
        <Transition name="fade" mode="out-in">
          <Icon v-if="loadingModel" icon="fa-solid fa-spinner" class="animate-spin"/>
          <div v-else class="flex gap-1 items-center">
            <div class="relative">
              <Icon icon="fa-solid fa-chart-column" class="cursor-pointer transition-all" :class="showLog ? 'scale-115' : 'hover:scale-115 text-gray-500'" @click="toggleChart"/>
              <div v-if="currentLog.epochs.length" v-show="showLog" class="absolute right-0 top-full z-50 bg-white shadow-lg rounded-sm w-[50vw]" >
                <h2 class="column-title">Training log chart</h2>
                <div class="flex gap-1 mb-6 overflow-x-auto p-2">
                  <button
                    v-for="(metric, i) in allMetricKeys"
                    :key="metric"
                    @click="toggleMetric(metric)"
                    class="text-sm mb-2"
                    :style="selectedMetrics.includes(metric)
                      ? { backgroundColor: CHART_COLORS[i % CHART_COLORS.length] + '22', borderColor: CHART_COLORS[i % CHART_COLORS.length], color: CHART_COLORS[i % CHART_COLORS.length] }
                      : { backgroundColor: '#f1f5f9', borderColor: '#cbd5e1', color: '#94a3b8' }"
                  >
                    {{ metric }}
                  </button>
                </div>
                <div style="height: 50vh">
                  <canvas ref="chartCanvas"></canvas>
                </div>
              </div>
            </div>
            <div class="relative">
              <Icon icon="fa-solid fa-table" class="cursor-pointer transition-all" :class="showMetricsTable ? 'scale-115' : 'hover:scale-115 text-gray-500'" @click="showMetricsTable = !showMetricsTable"/>
              <div v-if="currentMetricsTable" v-show="showMetricsTable" class="text-nowrap absolute right-0 max-h-[75vh] shadow-lg overflow-x-hidden z-10 text-black grid gap-x-1 rounded-b-sm">
                <div class="bg-white col-span-2 sticky top-0 grid grid-cols-2 gap-x-1">
                  <h2 class="column-title">Eval. test results (1 GT)</h2>
                  <h2 class="column-title">Eval. test results (N GT)</h2>
                </div>
                <div class="column bg-white p-2">
                  <div v-for="(val, metric) in (Object.values(currentMetricsTable.results_1)[0] ?? {})" :key="metric" >
                    {{ metric }}: {{ val }}
                  </div>
                </div>
                <div class="column bg-white p-2">
                  <div v-for="(val, metric) in (Object.values(currentMetricsTable.results_N)[0] ?? {})" :key="metric">
                    {{ metric }}: {{ val }}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Transition>
        <select :disabled="loadingModel" class="cursor-pointer" @change="(e) => setNewModel(e.target.value)">
          <option
          v-for="model in models"
          :key="model"
          :selected="model === currentModel"
          :value="model"
          >
          {{ model.split("/").pop()?.split(".")[0] }}
        </option>
      </select>
    </div>
  </h2>
    <div class="column">
      <h2 class="column-title">Available tracks</h2>
      <ul>
        <li v-if="tracks.length === 0" class="p-2 text-gray-400">
          No songs available
        </li>
        <li
          v-else
          v-for="track in tracks"
          @click="selectedTracks.push(track)"
          :key="track.id"
          class="group px-2 py-1 rounded-sm hover:bg-white bg-white/50"
        >
          {{ track.name }}
          <div class="breadcrumb-group">
            <div class="breadcrumb">ID: {{ track.id }}</div>
          </div>
        </li>
      </ul>
    </div>
    <div class="column">
      <h2 class="column-title">Listening history</h2>
      <ul>
        <li v-if="selectedTracks.length === 0" class="p-2 text-gray-400">
          No songs added yet
        </li>
        <li
          v-else
          v-for="(track, idx) in selectedTracks"
          :key="track.name"
          @click="selectedTracks.splice(idx, 1)"
          class="group"
        >
          <span class="rank">{{ idx + 1 }}</span>
          <div class="breadcrumb-group">
            <div class="breadcrumb">ID: {{ track.id }}</div>
          </div>
          {{ track.name }}
        </li>
      </ul>
    </div>
    <div class="column">
      <h2 class="column-title">Current recommendations</h2>
      <ul>
        <li v-if="recommendations.length === 0" class="p-2 text-gray-400">
          No recommendations yet
        </li>
        <li
          v-else
          v-for="recommendation in recommendations"
          :key="recommendation.track_id"
          @click="selectTrack(recommendation.track_id, recommendation.name)"
          class="group"
        >
          <span class="rank">
            {{ recommendation.rank }}
          </span>
          <div class="breadcrumb-group">
            <div class="breadcrumb">ID: {{ recommendation.track_id }}</div>
            <div class="breadcrumb">Score: {{ recommendation.score.toFixed(2) }}</div>
          </div>
          <span :class="grayedOut(recommendation.name)">
            {{ recommendation.name }}
          </span>
        </li>
      </ul>
    </div>
    <div class="flex gap-[3rem]">
      <div class="grid grid-cols-[min-content_3rem_min-content]">
        <button @click="prevPage">
          <Icon icon="fa-solid fa-chevron-left" />
        </button>
        <span class="m-auto">{{ page }}</span>
        <button @click="nextPage">
          <Icon icon="fa-solid fa-chevron-right" />
        </button>
      </div>
      <form class="flex flex-row gap-2 w-full">
        <input type="text" v-model="searchQuery" placeholder="Search..."/>
        <button type="submit" :disabled="isFetchingTracks" @click.prevent="resetPage(); getTracks();">
          <Icon icon="fa-solid fa-search" />
        </button>
      </form>
    </div>
    <div class="flex justify-end gap-2">
      <!-- <button @click="selectedTracks.splice(0, selectedTracks.length)">
        <Icon icon="fa-solid fa-eraser" />
      </button> -->
    </div>
    <form class="flex gap-2">
      <input v-model="k" placeholder="K..."/>
      <button type="submit" @click.prevent="getRecommendations">
        <Icon icon="fa-solid fa-thumbs-up" />
      </button>
      <input v-model="interval" placeholder="Interval (s)..."/>
      <button @click.prevent="autoPlaying = !autoPlaying; if (autoPlaying) autoPlay();">
        <Icon :icon="autoPlaying ? 'fa-solid fa-stop' : 'fa-solid fa-play'"/>
      </button>
      <button @click.prevent="autoContinue = !autoContinue">
        <Icon :icon="autoContinue ? 'fa-solid fa-clock-rotate-left' : 'fa-solid fa-rotate-left'"/>
      </button>
    </form>
  </main>
</template>