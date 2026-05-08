<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{
  tracks: Track[];
  before: number;
}>();

const emit = defineEmits<{
  (e: "add", track: Track): void;
}>();

const maxTimestamp: number = 1421745720;
const secondsInDay = 86400;

const date = computed(() => { 
  return new Date((maxTimestamp - (props.before - 1) * secondsInDay) * 1000) .toLocaleDateString();
});
</script>

<template>
  <div class="column">
    <h2 class="column-title flex justify-between items-center">
      <span>Available tracks</span>
      <span class="text-sm text-black/25">Before: {{ date }}</span>
    </h2>
    <TransitionGroup name="fade" tag="ul">
      <li v-if="tracks.length === 0" class="p-2 text-gray-400" key="empty">
        No songs available
      </li>
      <li
        v-else
        v-for="track in tracks"
        @click="emit('add', track)"
        :key="track.id"
        class="group track"
      >
        {{ track.name }}
        <div class="breadcrumb-group">
          <div class="breadcrumb">ID: {{ track.id }}</div>
        </div>
      </li>
    </TransitionGroup>
  </div>
</template>
