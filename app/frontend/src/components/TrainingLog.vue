<script setup lang="ts">
import Button from "./Button.vue";
import {
  CHART_COLORS,
  chartCanvas,
  toggleMetric,
  metricKeys,
  selectedMetrics,
} from "../chart";
</script>

<template>
  <div class="absolute right-0 top-full z-50 bg-white shadow-lg rounded-sm w-[50vw]">
    <h2 class="column-title">Training log chart</h2>
    <div class="flex gap-1 mb-6 overflow-x-auto p-2">
      <Button
        v-for="(metric, i) in metricKeys"
        :key="metric"
        @click="toggleMetric(metric)"
        class="text-sm mb-2"
        :style="
          selectedMetrics.includes(metric)
            ? {
                backgroundColor:
                  CHART_COLORS[i % CHART_COLORS.length] + '22',
                borderColor:
                  CHART_COLORS[i % CHART_COLORS.length],
                color: CHART_COLORS[i % CHART_COLORS.length],
              }
            : {
                backgroundColor: '#f1f5f9',
                borderColor: '#cbd5e1',
                color: '#94a3b8',
              }
        "
      >
        {{ metric }}
      </Button>
    </div>
    <div style="height: 50vh">
      <canvas ref="chartCanvas"></canvas>
    </div>
  </div>
</template>

<style scoped>
@reference "tailwindcss";


</style>