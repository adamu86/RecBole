<script setup lang="ts">
defineProps<{
  metrics: Metrics;
}>();
</script>

<template>
  <div
    class="text-nowrap absolute right-[calc(100%+0.25rem)] top-0 max-h-[75vh] shadow-lg overflow-x-hidden z-10 text-black grid gap-x-1 rounded-b-sm bg-gray-300"
  >
    <div class="bg-white col-span-2 sticky top-0 grid grid-cols-2 gap-x-1">
      <h2 class="column-title">Eval. test results (1 GT)</h2>
      <h2 class="column-title">Eval. test results (N GT)</h2>
    </div>
    <div class="column bg-white grid grid-cols-[1fr_auto_auto] border border-gray-300">
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
    <div class="column bg-white grid grid-cols-[1fr_auto_auto] border border-gray-300">
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
