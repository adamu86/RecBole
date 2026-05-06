<script setup lang="ts">
import { ref } from "vue";

const props = defineProps({
  info: { type: String, default: "" },
  disabled: { type: Boolean, default: false },
  icon: { type: String, default: "" },
  buttonStyle: { type: [Object, String], default: () => ({}) },
});

const tooltip = ref<HTMLElement | null>(null);
const align = ref<"left" | "center" | "right">("center");

const updateAlignment = () => {
  if (!tooltip.value) return;
  const rect = tooltip.value.getBoundingClientRect();
  if (rect.right > window.innerWidth) {
    align.value = "right";
  } else if (rect.left < 0) {
    align.value = "left";
  } else {
    align.value = "center";
  }
};
</script>

<template>
  <div
    :class="{ 'cursor-not-allowed': disabled }"
    class="relative group"
    @mouseenter="updateAlignment"
    @mouseleave="align = 'center'"
  >
    <button class="app-button" :disabled="disabled" :style="buttonStyle">
      <Icon v-if="icon" :icon="['fa-solid', 'fa-' + icon]" />
      <slot></slot>
    </button>
    <div
      v-if="props.info"
      ref="tooltip"
      class="absolute text-nowrap bottom-full mb-1 group-hover:block hidden breadcrumb transition-none!"
      :class="{
        'left-1/2 -translate-x-1/2': align === 'center',
        'left-0': align === 'left',
        'right-0': align === 'right',
      }"
    >
      {{ info }}
    </div>
  </div>
</template>