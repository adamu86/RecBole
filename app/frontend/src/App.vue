<script setup lang="ts">
import {
  fetchTracks,
  fetchRecommendations,
  fetchModels,
  setModel,
  fetchStatus,
  fetchMetrics,
  type TrackItem,
  type TrackRecommendation,
} from "./service";
import {
  CHART_COLORS,
  chartCanvas,
  buildChart,
  toggleChart,
  toggleMetric,
  metricKeys,
  selectedMetrics,
  currentLog,
  showLog,
  loadLog,
  type MetricsGroup,
  currentMetricsTable,
  showMetricsTable,
  getMetrics,
} from "./chart";
import { onMounted, watch, reactive } from "vue";

const availableTracks = reactive({
  searchQuery: "Avicii",
  items: [] as TrackItem[],
  page: {
    current: 1,
    itemsLimit: 25,
    get itemsOffset(): number {
      return (this.current - 1) * this.itemsLimit;
    },
    next() {
      this.current++;
    },
    prev() {
      if (this.current > 1) this.current--;
    },
    reset() {
      this.current = 1;
    },
  },
  isFetching: false as boolean,
  async fetch() {
    this.isFetching = true;
    try {
      const data = await fetchTracks(
        this.page.itemsOffset,
        this.page.itemsLimit,
        this.searchQuery,
      );
      if (data) this.items = data.tracks;
    } finally {
      this.isFetching = false;
    }
  },
});
watch(
  () => availableTracks.page.current,
  () => {
    availableTracks.fetch();
  },
);
const search = () => {
  availableTracks.page.reset();
  if (availableTracks.page.current === 1) availableTracks.fetch();
};

const listeningHistory = reactive({
  items: [] as TrackItem[],
  addTrack(track: TrackItem | TrackRecommendation) {
    this.items.push({
      id: track.id,
      name: track.name,
    });
  },
  removeTrack(track: TrackItem) {
    this.items = this.items.filter((t) => t.id !== track.id);
  },
});

const recommendations = reactive({
  topk: 25 as number,
  items: [] as TrackRecommendation[],
  interval: 5 as number,
  autoContinue: {
    value: false as boolean,
    toggle() {
      this.value = !this.value;
    },
  },
  autoPlay: {
    value: false as boolean,
    toggle() {
      this.value = !this.value;
    },
  },
  isFetching: false as boolean,
  async fetch() {
    this.isFetching = true;
    try {
      const data = await fetchRecommendations(
        listeningHistory.items.map((t: TrackItem) => parseInt(t.id)),
        recommendations.topk,
      );
      if (data) this.items = data.recommendations;
    } finally {
      this.isFetching = false;
    }
  },
});
watch(
  () => listeningHistory.items.length,
  () => {
    if (recommendations.autoContinue.value && !recommendations.autoPlay.value) {
      recommendations.fetch();
    }
  },
);
watch(
  () => recommendations.autoPlay.value,
  async (isActive) => {
    if (!isActive) return;
    while (recommendations.autoPlay.value) {
      await recommendations.fetch();
      const selectedIds = new Set(listeningHistory.items.map((t) => t.id));
      const next = recommendations.items.find((r) => !selectedIds.has(r.id));
      if (next) {
        listeningHistory.addTrack({
          id: next.id,
          name: next.name,
        });
      }
      await new Promise((resolve) =>
        setTimeout(resolve, recommendations.interval * 1000),
      );
      if (!recommendations.autoPlay.value) break;
    }
  },
);

const models = reactive({
  current: "" as string,
  items: [] as string[],
  isFetching: false as boolean,
  async fetch() {
    this.isFetching = true;
    try {
      const data = await fetchModels();
      if (data) this.items = data;
    } finally {
      this.isFetching = false;
    }
  },
  async setNewModel(modelPath: string) {
    this.isFetching = true;
    try {
      await setModel(modelPath);
    } finally {
      this.isFetching = false;
    }
  },
});
watch(
  () => models.current,
  async (newModelPath, oldModelPath) => {
    if (!newModelPath || !oldModelPath) return;
    showLog.value = false;
    showMetricsTable.value = false;
    availableTracks.items = [];
    availableTracks.page.reset();
    listeningHistory.items = [];
    recommendations.items = [];
    recommendations.autoContinue.value = false;
    recommendations.autoPlay.value = false;
    await models.setNewModel(newModelPath);
    await loadAll();
  },
);

const loadAll = async () => {
  await models.fetch();
  await availableTracks.fetch();
  await getMetrics();
  await loadLog();
  buildChart();
};

const init = async () => {
  const status = await fetchStatus();
  Object.entries(status).forEach(([key, value]) =>
    console.log(`${key}: ${value}`),
  );
  models.current = status.model;
  await loadAll();
};

onMounted(init);
</script>

<template>
  <main
    class="grid grid-rows-[auto_1fr_auto] grid-cols-3 gap-2 p-4 h-screen overflow-hidden"
    :class="models.isFetching ? 'pointer-events-none opacity-50' : ''"
  >
    <h2 class="col-span-3 mb-2 flex flex-row justify-between">
      <span class="text-3xl font-bold"> Music Recommender </span>
      <div v-if="models.items.length > 0" class="flex items-center">
        <Transition name="fade" mode="out-in">
          <Icon
            v-if="models.isFetching"
            icon="fa-solid fa-spinner"
            class="animate-spin"
          />
          <div v-else class="flex gap-1 items-center">
            <div class="relative">
              <Icon
                icon="fa-solid fa-chart-column"
                class="cursor-pointer transition-all"
                :class="showLog ? 'scale-115' : 'hover:scale-115 text-gray-500'"
                @click="toggleChart"
              />
              <div
                v-if="currentLog.epochs.length"
                v-show="showLog"
                class="absolute right-0 top-full z-50 bg-white shadow-lg rounded-sm w-[50vw]"
              >
                <h2 class="column-title">Training log chart</h2>
                <div class="flex gap-1 mb-6 overflow-x-auto p-2">
                  <button
                    v-for="(metric, i) in metricKeys"
                    :key="metric"
                    @click="toggleMetric(metric)"
                    class="text-sm mb-2"
                    :style="
                      selectedMetrics.includes(metric)
                        ? {
                            backgroundColor:
                              CHART_COLORS[i % CHART_COLORS.length] + '22',
                            borderColor: CHART_COLORS[i % CHART_COLORS.length],
                            color: CHART_COLORS[i % CHART_COLORS.length],
                          }
                        : {
                            backgroundColor: '#f1f5f9',
                            borderColor: '#cbd5e1',
                            color: '#94a3b8',
                          }
                    "
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
              <Icon
                icon="fa-solid fa-table"
                class="cursor-pointer transition-all"
                :class="
                  showMetricsTable
                    ? 'scale-115'
                    : 'hover:scale-115 text-gray-500'
                "
                @click="showMetricsTable = !showMetricsTable"
              />
              <div
                v-if="currentMetricsTable"
                v-show="showMetricsTable"
                class="text-nowrap absolute right-0 max-h-[75vh] shadow-lg overflow-x-hidden z-10 text-black grid gap-x-1 rounded-b-sm"
              >
                <div
                  class="bg-white col-span-2 sticky top-0 grid grid-cols-2 gap-x-1"
                >
                  <h2 class="column-title">Eval. test results (1 GT)</h2>
                  <h2 class="column-title">Eval. test results (N GT)</h2>
                </div>
                <div class="column bg-white grid grid-cols-[1fr_auto_auto]">
                  <template
                    v-for="(val, metric) in Object.values(
                      currentMetricsTable.results_1,
                    )[0] ?? {}"
                    :key="metric"
                  >
                    <div class="text-right px-2 py-1">
                      {{ metric.split("@")[0] }}
                    </div>
                    <div class="px-2 py-1 border-x-[1px] border-x-gray-300">
                      @{{ metric.split("@")[1] }}
                    </div>
                    <div class="px-2 py-1 font-medium text-left">
                      {{ val.toFixed(3) }}
                    </div>
                  </template>
                </div>
                <div class="column bg-white grid grid-cols-[1fr_auto_auto]">
                  <template
                    v-for="(val, metric) in Object.values(
                      currentMetricsTable.results_N,
                    )[0] ?? {}"
                    :key="metric"
                  >
                    <div class="text-right px-2 py-1">
                      {{ metric.split("@")[0] }}
                    </div>
                    <div class="px-2 py-1 border-x-[1px] border-x-gray-300">
                      @{{ metric.split("@")[1] }}
                    </div>
                    <div class="px-2 py-1 font-medium text-left">
                      {{ val.toFixed(3) }}
                    </div>
                  </template>
                </div>
              </div>
            </div>
          </div>
        </Transition>
        <select
          :disabled="models.isFetching"
          class="cursor-pointer"
          @change="
            (e) => (models.current = (e.target as HTMLSelectElement).value)
          "
        >
          <option
            v-for="model in models.items"
            :key="model"
            :selected="model === models.current"
            :value="model"
          >
            {{ model.split("/")[1]?.split("_")[0] }} :
            {{ model.split("/")[1]?.split("__")[1]?.split("_")[0] }}
          </option>
        </select>
      </div>
    </h2>
    <div class="column">
      <h2 class="column-title">Available tracks</h2>
      <TransitionGroup name="fade" tag="ul">
        <li v-if="availableTracks.items.length === 0" class="p-2 text-gray-400" key="empty">
          No songs available
        </li>
        <li
          v-else
          v-for="track in availableTracks.items"
          @click="listeningHistory.addTrack(track)"
          :key="track.id"
          class="group px-2 py-1 rounded-sm hover:bg-white bg-white/50"
        >
          {{ track.name }}
          <div class="breadcrumb-group">
            <div class="breadcrumb">ID: {{ track.id }}</div>
          </div>
        </li>
      </TransitionGroup>
    </div>
    <div class="column">
      <h2 class="column-title">Listening history</h2>
      <TransitionGroup name="fade" tag="ul">
        <li
          v-if="listeningHistory.items.length === 0"
          class="p-2 text-gray-400"
          key="empty"
        >
          No songs added yet
        </li>
        <li
          v-else
          v-for="(track, idx) in listeningHistory.items"
          :key="track.name"
          @click="listeningHistory.removeTrack(track)"
          class="group"
        >
          <span class="rank">{{ idx + 1 }}</span>
          <div class="breadcrumb-group">
            <div class="breadcrumb">ID: {{ track.id }}</div>
          </div>
          {{ track.name }}
        </li>
      </TransitionGroup>
    </div>
    <div class="column">
      <h2 class="column-title">Current recommendations</h2>
      <TransitionGroup name="fade" tag="ul">
        <li v-if="recommendations.items.length === 0" class="p-2 text-gray-400" key="empty">
          No recommendations yet
        </li>
        <li
          v-else
          v-for="recommendation in recommendations.items"
          :key="recommendation.id"
          @click="listeningHistory.addTrack(recommendation)"
          class="group"
        >
          <span class="rank">
            {{ recommendation.rank }}
          </span>
          <div class="breadcrumb-group">
            <div class="breadcrumb">ID: {{ recommendation.id }}</div>
            <div class="breadcrumb">
              Score: {{ recommendation.score.toFixed(2) }}
            </div>
          </div>
          <span
            :class="
              listeningHistory.items.some((t) => t.id === recommendation.id)
                ? 'opacity-25'
                : ''
            "
          >
            {{ recommendation.name }}
          </span>
        </li>
      </TransitionGroup>
    </div>
    <div class="flex gap-[3rem]">
      <div class="grid grid-cols-[min-content_3rem_min-content]">
        <button @click="availableTracks.page.prev()">
          <Icon icon="fa-solid fa-chevron-left" />
        </button>
        <span class="m-auto">{{ availableTracks.page.current }}</span>
        <button @click="availableTracks.page.next()">
          <Icon icon="fa-solid fa-chevron-right" />
        </button>
      </div>
      <form class="flex flex-row gap-2 w-full">
        <input
          type="text"
          v-model="availableTracks.searchQuery"
          placeholder="Search..."
        />
        <button
          type="submit"
          :disabled="availableTracks.isFetching"
          @click.prevent="search()"
        >
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
      <input v-model="recommendations.topk" placeholder="K..." />
      <button type="submit" @click.prevent="recommendations.fetch()">
        <Icon icon="fa-solid fa-thumbs-up" />
      </button>
      <input v-model="recommendations.interval" placeholder="Interval (s)..." />
      <button @click.prevent="recommendations.autoPlay.toggle()">
        <Icon
          :icon="
            recommendations.autoPlay.value
              ? 'fa-solid fa-stop'
              : 'fa-solid fa-play'
          "
        />
      </button>
      <button
        @click.prevent="recommendations.autoContinue.toggle()"
        :disabled="recommendations.autoPlay.value"
        :class="recommendations.autoPlay.value ? 'cursor-not-allowed' : ''"
      >
        <Icon
          :icon="
            recommendations.autoContinue.value
              ? 'fa-solid fa-clock-rotate-left'
              : 'fa-solid fa-rotate-left'
          "
        />
      </button>
    </form>
  </main>
</template>
