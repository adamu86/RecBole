<script setup lang="ts">
const modelValue = defineModel<string | number>();
</script>

<template>
  <input class="app-input" v-model="modelValue" v-bind="$attrs" type="text" />
</template>
