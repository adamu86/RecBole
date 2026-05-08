<script setup lang="ts">
import {
  CHART_COLORS,
  chartCanvas,
  toggleMetric,
  metricKeys,
  selectedMetrics,
} from "../chart";
import { computed } from "vue";

const kCount = computed(() => {
  const ks = metricKeys.value.filter(m => m.includes('@')).map(m => m.split('@')[1]);
  return new Set(ks).size || 4;
});
</script>

<template>
  <div class="absolute right-[calc(100%+0.25rem)] top-0 z-50 bg-gray-100 shadow-lg rounded-sm w-fit">
    <h2 class="column-title">Training log chart</h2>
    <div class="px-2 pt-2">
      <div class="flex flex-wrap gap-1 mb-1">
        <button
          v-for="metric in metricKeys.filter(m => !m.includes('@'))"
          :key="metric"
          @click="toggleMetric(metric)"
          class="px-3 py-1 text-xs font-semibold rounded-full border transition-all cursor-pointer whitespace-nowrap"
          :class="
            selectedMetrics.includes(metric)
              ? ''
              : 'bg-white border-gray-200 text-gray-500 hover:bg-gray-50 hover:text-gray-700'
          "
          :style="
            selectedMetrics.includes(metric)
              ? {
                  backgroundColor: CHART_COLORS[metricKeys.indexOf(metric) % CHART_COLORS.length] + '22',
                  borderColor: CHART_COLORS[metricKeys.indexOf(metric) % CHART_COLORS.length],
                  color: CHART_COLORS[metricKeys.indexOf(metric) % CHART_COLORS.length],
                }
              : {}
          "
        >
          {{ metric }}
        </button>
      </div>

      <div 
        class="grid grid-flow-col gap-1 pb-2 max-w-full overflow-x-auto overflow-y-hidden"
        :style="{ gridTemplateRows: `repeat(${kCount}, minmax(0, 1fr))` }"
      >
        <button
          v-for="metric in metricKeys.filter(m => m.includes('@'))"
          :key="metric"
          @click="toggleMetric(metric)"
          class="px-3 py-1 text-xs font-semibold rounded-full border transition-all cursor-pointer whitespace-nowrap"
          :class="
            selectedMetrics.includes(metric)
              ? ''
              : 'bg-white border-gray-200 text-gray-500 hover:bg-gray-50 hover:text-gray-700'
          "
          :style="
            selectedMetrics.includes(metric)
              ? {
                  backgroundColor: CHART_COLORS[metricKeys.indexOf(metric) % CHART_COLORS.length] + '22',
                  borderColor: CHART_COLORS[metricKeys.indexOf(metric) % CHART_COLORS.length],
                  color: CHART_COLORS[metricKeys.indexOf(metric) % CHART_COLORS.length],
                }
              : {}
          "
        >
          {{ metric }}
        </button>
      </div>
    </div>
    <div style="height: 50vh">
      <canvas ref="chartCanvas"></canvas>
    </div>
  </div>
</template>

<style scoped>
@reference "tailwindcss";


</style>