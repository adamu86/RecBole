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
        v-for="(recommendation, idx) in recommendations"
        :key="recommendation.id"
        @click="emit('add', recommendation)"
        class="group track"
      >
        <span class="rank">
          <!-- {{ recommendation.rank }} -->
            {{ idx + 1 }}
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
          <!-- <div v-if="recommendation.tags?.length" class="flex flex-wrap gap-1 mt-2">
            <span v-for="tag in recommendation.tags" :key="tag" class="px-1.5 py-0.5 text-xs bg-gray-300 text-gray-700 rounded">
              {{ tag }}
            </span>
          </div> -->
        </span>
      </li>
    </TransitionGroup>
  </div>
</template>
