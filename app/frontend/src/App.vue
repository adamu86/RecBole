<script setup lang="ts">
import {
  fetchTracks,
  fetchRecommendations,
  fetchModels,
  setModel,
  fetchStatus,
  type TrackItem,
  type TrackRecommendation,
} from "./service";
import { ref, computed, onMounted, watch } from "vue";

const page = ref<number>(1);
const limit = ref<number>(25);
const offset = computed(() => (page.value - 1) * limit.value);
const searchQuery = ref<string>("Avicii");
const k = ref<number>(20);
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

const init = async () => {
  const status = await fetchStatus();
  currentModel.value = status.model;
  console.log(status);
  getModels();
  getTracks();
}

watch(() => selectedTracks.value.length, () => {
  if (autoContinue.value && !autoPlaying.value) {
    getRecommendations();
  }
});

onMounted(init);
</script>

<template>
  <main class="grid grid-rows-[auto_1fr_auto] grid-cols-3 gap-4 p-4 h-screen overflow-hidden" :class="loadingModel ? '[&>*]:cursor-not-allowed' : ''">
    <h2 class="col-span-3 flex flex-row justify-between">
      <span class="text-3xl font-bold">
        Music Recommender
      </span>
      <div class="flex items-center">
        <Icon icon="fa-solid fa-spinner" v-if="loadingModel" class="animate-spin"/>
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
      <h2 class="column-title">Available Tracks</h2>
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
      <h2 class="column-title">Listening History</h2>
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
      <h2 class="column-title">Current Recommendations</h2>
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
      <button @click="selectedTracks.splice(0, selectedTracks.length)">
        <Icon icon="fa-solid fa-eraser" />
      </button>
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