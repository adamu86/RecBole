<script setup lang="ts">
import {
  fetchTracks,
  fetchRecommendations,
  type TrackItem,
  type TrackRecommendation,
} from "./service";
import { ref, computed, onMounted } from "vue";

const page = ref<number>(1);
const limit = ref<number>(25);
const offset = computed(() => (page.value - 1) * limit.value);
const searchQuery = ref<string>("Avicii");
const k = ref<number>(20);
const interval = ref<number>(2);
const autoPlaying = ref<boolean>(false);
const selectedTracks = ref<TrackItem[]>([]);
const tracks = ref<TrackItem[]>([]);
const recommendations = ref<TrackRecommendation[]>([]);

const getTracks = async () => {
  const response = await fetchTracks(
    offset.value,
    limit.value,
    searchQuery.value,
  );
  tracks.value = response.tracks;
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

onMounted(() => {
  getTracks();
});
</script>

<template>
  <main class="grid grid-rows-[auto_1fr] gap-4 p-4 h-screen overflow-hidden">
    <h2 class="px-2 text-2xl font-bold">Music Recommender</h2>
    <div class="grid grid-cols-3 gap-4 min-h-0">
      <div class="grid grid-rows-[auto_1fr] min-h-0">
        <h2 class="p-2 text-xl">Available Tracks</h2>
        <ul
          class="flex flex-col gap-1 bg-black/5 p-2 rounded-sm overflow-y-auto"
        >
          <li
            @click="selectedTracks.push(track)"
            v-for="track in tracks"
            :key="track.id"
            class="group relative px-2 py-1 rounded-sm hover:bg-white cursor-pointer"
          >
            {{ track.name }}
            <div
              class="absolute top-[calc(100%+0.25rem)] left-0 z-100 grid grid-cols-[auto_auto] gap-1 pointer-events-none"
            >
              <div
                class="bg-teal-600 text-white rounded px-1 opacity-0 group-hover:opacity-100"
              >
                ID: {{ track.id }}
              </div>
            </div>
          </li>
        </ul>
      </div>
      <div class="grid grid-rows-[auto_1fr] min-h-0">
        <h2 class="p-2 text-xl">Listening History</h2>
        <ul
          class="flex flex-col gap-1 overflow-y-auto p-2 bg-black/5 rounded-sm"
        >
          <li
            v-for="(track, idx) in selectedTracks"
            :key="track.name"
            @click="selectedTracks.splice(idx, 1)"
            class="group relative cursor-pointer"
          >
            <span class="bg-white rounded px-1 mr-0.5">
              {{ idx + 1 }}
            </span>
            <div
              class="absolute top-full left-0 z-100 grid grid-cols-[auto_auto] gap-1 pointer-events-none"
            >
              <div
                class="bg-teal-600 text-white rounded px-1 opacity-0 group-hover:opacity-100"
              >
                ID: {{ track.id }}
              </div>
            </div>
            {{ track.name }}
          </li>
        </ul>
      </div>
      <div class="grid grid-rows-[auto_1fr] min-h-0">
        <h2 class="p-2 text-xl">Current Recommendations</h2>
        <ul
          class="flex flex-col gap-1 overflow-y-auto p-2 bg-black/5 rounded-sm"
        >
          <li
            class="group relative cursor-pointer"
            v-for="recommendation in recommendations"
            :key="recommendation.track_id"
            @click="
              selectedTracks.push({
                id: recommendation.track_id.toString(),
                name: recommendation.name,
              })
            "
          >
            <span class="bg-white rounded px-1 mr-0.5">
              {{ recommendation.rank }}
            </span>
            <div
              class="absolute top-full left-0 z-100 grid grid-cols-[auto_auto] gap-1 pointer-events-none"
            >
              <div
                class="bg-teal-600 text-white rounded px-1 opacity-0 group-hover:opacity-100"
              >
                ID: {{ recommendation.track_id }}
              </div>
              <div
                class="bg-teal-600 text-white rounded px-1 opacity-0 group-hover:opacity-100"
              >
                Score: {{ recommendation.score.toFixed(2) }}
              </div>
            </div>
            {{ recommendation.name }}
          </li>
        </ul>
      </div>
    </div>
    <div class="grid grid-cols-3 gap-4">
      <div class="flex flex-row gap-2">
        <div class="grid grid-cols-[auto_2rem_auto] gap-2 my-auto w-fit">
          <button
            class="h-10 p-2 aspect-square rounded-full bg-teal-600 hover:bg-teal-700 text-white cursor-pointer"
            @click="prevPage"
          >
            <Icon icon="fa-solid fa-chevron-left" />
          </button>
          <span class="p-2 mx-auto">{{ page }}</span>
          <button
            class="h-10 p-2 aspect-square rounded-full bg-teal-600 hover:bg-teal-700 text-white cursor-pointer"
            @click="nextPage"
          >
            <Icon icon="fa-solid fa-chevron-right" />
          </button>
        </div>
        <div class="flex flex-row gap-2 ml-auto">
          <input
            class="p-2 rounded-full border-2 border-gray-400 outline-none"
            type="text"
            v-model="searchQuery"
            placeholder="Search..."
          />
          <button
            class="aspect-square p-2 rounded-full bg-teal-600 hover:bg-teal-700 text-white cursor-pointer"
            @click="
              resetPage();
              getTracks();
            "
          >
            <Icon icon="fa-solid fa-search" />
          </button>
        </div>
      </div>
      <div></div>
      <div class="flex gap-2 my-auto">
        <input
          class="w-30 p-2 rounded-full border-2 border-gray-400 outline-none ml-auto"
          v-model="k"
          placeholder="K..."
        />
        <button
          class="aspect-square h-10 p-2 rounded-full bg-teal-600 hover:bg-teal-700 text-white cursor-pointer"
          @click="getRecommendations"
        >
          <Icon icon="fa-solid fa-thumbs-up" />
        </button>
        <input
          class="w-30 p-2 rounded-full border-2 border-gray-400 outline-none"
          v-model="interval"
          placeholder="Interval (s)..."
        />
        <button
          class="aspect-square h-10 p-2 rounded-full text-white cursor-pointer"
          :class="
            autoPlaying
              ? 'bg-rose-600 hover:bg-rose-700'
              : 'bg-teal-600 hover:bg-teal-700'
          "
          @click="
            autoPlaying = !autoPlaying;
            if (autoPlaying) autoPlay();
          "
        >
          <Icon
            class="mx-auto"
            :icon="autoPlaying ? 'fa-solid fa-stop' : 'fa-solid fa-play'"
          />
        </button>
      </div>
    </div>
  </main>
</template>
