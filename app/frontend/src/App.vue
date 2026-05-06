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
import { onMounted, onUnmounted, watch, reactive, computed } from "vue";

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
    if (
      recommendations.autoContinue.value &&
      !recommendations.autoPlay.value &&
      listeningHistory.items.length > 0
    ) {
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

const reloadPage = () => {
  window.location.reload();
};

const handleKeyDown = (e: KeyboardEvent) => {
  if (e.ctrlKey && e.key === "f") {
    e.preventDefault();
    document.getElementById("searchInput")?.focus();
  } else if (
    e.key === "Enter" &&
    document.activeElement?.id === "searchInput"
  ) {
    search();
  } else if (e.key === "Enter") {
    recommendations.fetch();
  }
};

onMounted(() => {
  init();
  window.addEventListener("keydown", handleKeyDown);
  document
    .getElementById("searchInput")
    ?.addEventListener("keydown", handleKeyDown);
});

onUnmounted(() => {
  window.removeEventListener("keydown", handleKeyDown);
  document
    .getElementById("searchInput")
    ?.removeEventListener("keydown", handleKeyDown);
});
</script>

<template>
  <main :class="models.isFetching ? 'pointer-events-none opacity-50' : ''">
    <h2 class="header">
      <span class="text-3xl font-bold italic flex gap-1.5" @click="reloadPage">
        <span class="">MusicRec</span>
        <Icon icon="fa-solid fa-music" class="mb-auto -skew-x-3 rotate-12" />
      </span>
      <div v-if="models.items.length > 0" class="flex items-center gap-2">
        <Transition name="fade" mode="out-in">
          <Button
            v-if="models.isFetching"
            icon="spinner"
            class="animate-spin! [&>button]:bg-transparent! [&>button>svg]:text-gray-500!"
          />
          <div v-else class="flex gap-2 items-center">
            <div class="relative">
              <Button @click="toggleChart" icon="chart-column" />
              <Transition name="slide-fade-top">
                <TrainingLog v-if="currentLog.epochs.length" v-show="showLog" />
              </Transition>
            </div>
            <div class="relative">
              <Button @click="toggleMetricsTable" icon="table" />
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
        <Button
          @click="availableTracks.page.prev()"
          info="Previous page"
          icon="chevron-left"
        />
        <span class="m-auto">{{ availableTracks.page.current }}</span>
        <Button
          @click="availableTracks.page.next()"
          info="Next page"
          icon="chevron-right"
        />
      </div>
      <div class="flex gap-2 w-1/2 ml-auto">
        <Input
          id="searchInput"
          v-model="availableTracks.searchQuery"
          placeholder="Search..."
        />
        <Button
          :disabled="availableTracks.isFetching"
          @click.prevent="search()"
          info="Search"
          icon="search"
        />
      </div>
    </div>
    <div class="flex gap-2">
      <!-- <Button
        @click="listeningHistory.items.splice(0, listeningHistory.items.length)"
        icon="eraser"
      /> -->
    </div>
    <div class="flex gap-2">
      <Input v-model="recommendations.topk" placeholder="K..." />
      <Button
        @click.prevent="recommendations.fetch()"
        :disabled="
          listeningHistory.items.length === 0 ||
          (recommendations.autoContinue.value &&
            recommendations.items.length !== 0) ||
          recommendations.autoPlay.value
        "
        info="Recommend"
        icon="thumbs-up"
      />
      <Button
        @click.prevent="recommendations.autoContinue.toggle()"
        :disabled="recommendations.autoPlay.value"
        :class="recommendations.autoPlay.value ? 'cursor-not-allowed' : ''"
        :info="
          recommendations.autoContinue.value
            ? 'Auto-continue mode'
            : 'Manual mode'
        "
        :icon="recommendations.autoContinue.value ? 'repeat' : 'hand-pointer'"
      />
      <Input v-model="recommendations.interval" placeholder="Interval (s)..." />
      <Button
        @click.prevent="recommendations.autoPlay.toggle()"
        :disabled="
          listeningHistory.items.length === 0 ||
          recommendations.autoContinue.value
        "
        :info="
          recommendations.autoPlay.value
            ? 'Auto-play is ON'
            : 'Auto-play is OFF'
        "
        :icon="recommendations.autoPlay.value ? 'stop' : 'play'"
      />
    </div>
  </main>
</template>
