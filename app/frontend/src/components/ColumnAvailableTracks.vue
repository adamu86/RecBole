<script setup lang="ts">
defineProps<{
  tracks: Track[];
}>();

const emit = defineEmits<{
  (e: "add", track: Track): void;
}>();
</script>

<template>
  <div class="column">
    <h2 class="column-title">Available tracks</h2>
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
