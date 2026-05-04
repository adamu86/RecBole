<script setup lang="ts">
defineProps<{
  recommendations: Track[];
  history: Track[];
}>();

const emit = defineEmits<{
  (e: "add", track: Track): void;
}>();
</script>

<template>
  <div class="column">
    <h2 class="column-title">Current recommendations</h2>
    <TransitionGroup name="fade" tag="ul">
      <li
        v-if="recommendations.length === 0"
        class="p-2 text-gray-400"
        key="empty"
      >
        No recommendations yet
      </li>
      <li
        v-else
        v-for="recommendation in recommendations"
        :key="recommendation.id"
        @click="emit('add', recommendation)"
        class="group track"
      >
        <span class="rank">
          {{ recommendation.rank }}
        </span>
        <div class="breadcrumb-group">
          <div class="breadcrumb">ID: {{ recommendation.id }}</div>
          <div class="breadcrumb">
            Score: {{ recommendation.score?.toFixed(2) }}
          </div>
        </div>
        <span
          :class="
            history.some((t) => t.id === recommendation.id) ? 'opacity-25' : ''
          "
        >
          {{ recommendation.name }}
        </span>
      </li>
    </TransitionGroup>
  </div>
</template>
