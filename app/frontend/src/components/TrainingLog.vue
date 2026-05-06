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
  <div class="absolute right-[calc(100%+0.25rem)] top-0 z-50 bg-white shadow-lg rounded-sm w-[50vw]">
    <h2 class="column-title">Training log chart</h2>
    <div class="flex gap-1 mb-6 overflow-x-auto p-2">
      <Button
        v-for="(metric, i) in metricKeys"
        :key="metric"
        @click="toggleMetric(metric)"
        class="mb-2"
        :buttonStyle="
          selectedMetrics.includes(metric)
            ? {
                backgroundColor:
                  CHART_COLORS[i % CHART_COLORS.length] + '22 !important',
                borderColor:
                  CHART_COLORS[i % CHART_COLORS.length] + ' !important',
                color: CHART_COLORS[i % CHART_COLORS.length] + ' !important',
              }
            : {
                backgroundColor: '#f1f5f9 !important',
                borderColor: '#cbd5e1 !important',
                color: '#94a3b8 !important',
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