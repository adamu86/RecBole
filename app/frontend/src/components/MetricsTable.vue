<script setup lang="ts">
defineProps<{
  metrics: Metrics;
}>();
</script>

<template>
  <div
    class="text-nowrap absolute right-[calc(100%+0.25rem)] top-0 w-max max-h-[75vh] shadow-lg overflow-x-hidden z-50 gap-x-1 text-black grid grid-cols-2 rounded-b-sm"
  >
    <div class="bg-gray-200 col-span-2 sticky top-0 grid grid-cols-2 gap-x-1">
      <h2 class="column-title">Eval. test results (1 GT)</h2>
      <h2 class="column-title">Eval. test results (N GT)</h2>
    </div>
    <div class="column bg-gray-100 grid grid-cols-[1fr_auto_auto] border border-gray-300">
      <template
        v-for="(val, metric) in Object.values(metrics.results_1)[0] ?? {}"
        :key="metric"
      >
        <div class="text-right px-2 py-1">
          {{ metric.split("@")[0] }}
        </div>
        <div class="px-2 py-1 border-x-[1px] border-x-gray-300">
          @{{ metric.split("@")[1] }}
        </div>
        <div class="px-2 py-1 font-medium text-left">
          {{ val.toFixed(3) }}
        </div>
      </template>
    </div>
    <div class="column bg-gray-100 grid grid-cols-[1fr_auto_auto] border border-gray-300">
      <template
        v-for="(val, metric) in Object.values(metrics.results_N)[0] ?? {}"
        :key="metric"
      >
        <div class="text-right px-2 py-1">
          {{ metric.split("@")[0] }}
        </div>
        <div class="px-2 py-1 border-x-[1px] border-x-gray-300">
          @{{ metric.split("@")[1] }}
        </div>
        <div class="px-2 py-1 font-medium text-left">
          {{ val.toFixed(3) }}
        </div>
      </template>
    </div>
  </div>
</template>
