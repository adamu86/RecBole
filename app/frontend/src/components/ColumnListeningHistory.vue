<script setup lang="ts">
defineProps<{
  tracks: Track[];
}>();

const emit = defineEmits<{
  (e: "remove", idx: number): void;
}>();
</script>

<template>
  <div class="column">
    <h2 class="column-title">Listening history</h2>
    <TransitionGroup name="fade" tag="ul">
      <li v-if="tracks.length === 0" class="p-2 text-gray-400" key="empty">
        No songs added yet
      </li>
      <li
        v-else
        v-for="(track, idx) in tracks"
        :key="track.name"
        @click="emit('remove', idx)"
        class="group track"
      >
        <span class="rank">{{ idx + 1 }}</span>
        <div class="breadcrumb-group">
          <div class="breadcrumb">ID: {{ track.id }}</div>
        </div>
        {{ track.name }}
      </li>
    </TransitionGroup>
  </div>
</template>
