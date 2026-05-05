<script setup lang="ts">
import {
  fetchTracks,
  fetchRecommendations,
  fetchModels,
  setModel,
  fetchStatus,
  fetchMetrics,
} from "./service";
import {
  buildChart,
  toggleChart,
  toggleMetricsTable,
  currentLog,
  showLog,
  loadLog,
  currentMetricsTable,
  showMetricsTable,
  getMetrics,
} from "./chart";
import TrainingLog from "./components/TrainingLog.vue";
import MetricsTable from "./components/MetricsTable.vue";
import Select from "./components/Select.vue";
import Input from "./components/Input.vue";
import Button from "./components/Button.vue";
import AvailableTracksColumn from "./components/ColumnAvailableTracks.vue";
import ListeningHistoryColumn from "./components/ColumnListeningHistory.vue";
import RecommendationsColumn from "./components/ColumnRecommendations.vue";
import { onMounted, watch, reactive, computed } from "vue";

const availableTracks = reactive({
  searchQuery: "Avicii",
  items: [] as Track[],
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
  items: [] as Track[],
  addTrack(track: Track) {
    this.items.push(track);
  },
  removeTrack(idx: number) {
    this.items.splice(idx, 1);
  },
});

const recommendations = reactive({
  topk: 25 as number,
  items: [] as Track[],
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
        listeningHistory.items.map((t: Track) => parseInt(t.id)),
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
      const nextTrack = recommendations.items.find(
        (r) => !selectedIds.has(r.id),
      );
      if (nextTrack) {
        listeningHistory.addTrack(nextTrack);
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
    await load();
  },
);

const modelOptions = computed(() =>
  models.items.map((model) => ({
    value: model,
    label: `${model.split("/")[1]?.split("_")[0]} : ${model.split("/")[1]?.split("__")[1]?.split("_")[0]}`,
  })),
);

const load = async () => {
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
  await load();
};

const scaleOnHover = (isActive: boolean) => {
  return isActive ? "scale-115" : "hover:scale-115 text-gray-500";
};

onMounted(init);
</script>

<template>
  <main :class="models.isFetching ? 'pointer-events-none opacity-50' : ''">
    <h2 class="header">
      <span class="text-3xl font-bold"> Music Recommender </span>
      <div v-if="models.items.length > 0" class="flex items-center gap-1.5">
        <Transition name="fade" mode="out-in">
          <Icon
            v-if="models.isFetching"
            icon="fa-solid fa-spinner"
            class="animate-spin"
          />
          <div v-else class="flex gap-1.5 items-center">
            <div class="relative">
              <Icon
                icon="fa-solid fa-chart-column"
                :class="['text-lg', scaleOnHover(showLog)]"
                @click="toggleChart"
              />
              <Transition name="slide-fade-top">
                <TrainingLog v-if="currentLog.epochs.length" v-show="showLog" />
              </Transition>
            </div>
            <div class="relative">
              <Icon
                icon="fa-solid fa-table"
                :class="['text-lg', scaleOnHover(showMetricsTable)]"
                @click="toggleMetricsTable"
              />
              <Transition name="slide-fade-top">
                <MetricsTable
                  v-if="currentMetricsTable"
                  v-show="showMetricsTable"
                  :metrics="currentMetricsTable"
                />
              </Transition>
            </div>
          </div>
        </Transition>
        <Select
          v-model="models.current"
          :options="modelOptions"
          :disabled="models.isFetching"
        />
      </div>
    </h2>
    <AvailableTracksColumn
      :tracks="availableTracks.items"
      @add="listeningHistory.addTrack($event)"
    />
    <ListeningHistoryColumn
      :tracks="listeningHistory.items"
      @remove="listeningHistory.removeTrack($event)"
    />
    <RecommendationsColumn
      :recommendations="recommendations.items"
      :history="listeningHistory.items"
      @add="listeningHistory.addTrack($event)"
    />
    <div class="flex gap-2">
      <div class="grid grid-cols-[min-content_3rem_min-content]">
        <Button @click="availableTracks.page.prev()">
          <Icon icon="fa-solid fa-chevron-left" />
        </Button>
        <span class="m-auto">{{ availableTracks.page.current }}</span>
        <Button @click="availableTracks.page.next()">
          <Icon icon="fa-solid fa-chevron-right" />
        </Button>
      </div>
      <div class="flex gap-2 w-1/2 ml-auto">
        <Input v-model="availableTracks.searchQuery" placeholder="Search..." />
        <Button
          :disabled="availableTracks.isFetching"
          @click.prevent="search()"
        >
          <Icon icon="fa-solid fa-search" />
        </Button>
      </div>
    </div>
    <div class="flex gap-2">
      <!-- <Button @click="selectedTracks.splice(0, selectedTracks.length)">
        <Icon icon="fa-solid fa-eraser" />
      </Button> -->
    </div>
    <div class="flex gap-2">
      <Input v-model="recommendations.topk" placeholder="K..." />
      <Button @click.prevent="recommendations.fetch()">
        <Icon icon="fa-solid fa-thumbs-up" />
      </Button>
      <Input v-model="recommendations.interval" placeholder="Interval (s)..." />
      <Button @click.prevent="recommendations.autoPlay.toggle()">
        <Icon
          :icon="
            recommendations.autoPlay.value
              ? 'fa-solid fa-stop'
              : 'fa-solid fa-play'
          "
        />
      </Button>
      <Button
        @click.prevent="recommendations.autoContinue.toggle()"
        :disabled="recommendations.autoPlay.value"
        :class="recommendations.autoPlay.value ? 'cursor-not-allowed' : ''"
      >
        <Icon
          :icon="
            recommendations.autoContinue.value
              ? 'fa-solid fa-repeat'
              : 'fa-solid fa-hand-pointer'
          "
        />
      </Button>
    </div>
  </main>
</template>
