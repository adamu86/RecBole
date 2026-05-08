<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch, nextTick } from "vue";

const props = defineProps<{
  options: { label: string; value: string }[];
  placeholder?: string;
  disabled?: boolean;
}>();

const modelValue = defineModel<string>();

const isOpen = ref(false);
const selectRef = ref<HTMLElement | null>(null);

const toggle = () => {
  if (props.disabled) return;
  isOpen.value = !isOpen.value;
};

const selectOption = (value: string) => {
  modelValue.value = value;
  isOpen.value = false;
};

const selectedLabel = computed(() => {
  const selected = props.options.find((o) => o.value === modelValue.value);
  return selected ? selected.label : props.placeholder || "Select...";
});

const longestLabel = computed(() => {
  let max = props.placeholder || "Select...";
  for (const option of props.options) {
    if (option.label.length > max.length) {
      max = option.label;
    }
  }
  return max;
});

const handleClickOutside = (event: MouseEvent) => {
  if (selectRef.value && !selectRef.value.contains(event.target as Node)) {
    isOpen.value = false;
  }
};

watch(isOpen, (newValue) => {
  if (!newValue) {
    (document.activeElement as HTMLElement).blur();
  }
});

onMounted(() => {
  document.addEventListener("click", handleClickOutside);
});

onUnmounted(() => {
  document.removeEventListener("click", handleClickOutside);
});
</script>

<template>
  <div class="relative w-max" ref="selectRef">
    <div
      id="select"
      tabindex="0"
      class="app-input cursor-pointer"
      @click="toggle"
    >
      <div class="grid items-center">
        <span
          class="invisible col-start-1 row-start-1 whitespace-nowrap"
          aria-hidden="true"
        >
          {{ longestLabel }}
        </span>
        <span class="truncate font-medium col-start-1 row-start-1">
          {{ selectedLabel }}
        </span>
      </div>
      <Icon
        icon="fa-solid fa-chevron-down"
        class="ml-2 text-sm! text-primary/75 text-sm flex"
        :class="{ '-rotate-x-180': isOpen }"
      />
    </div>

    <Transition name="slide-fade-top">
      <div
        v-if="isOpen"
        class="ring-1 ring-input/20 absolute z-50 w-full p-1 mt-2 bg-gray-100 rounded-base shadow-lg max-h-64 overflow-y-auto flex flex-col p-1"
      >
        <div
          v-for="option in options"
          :key="option.value"
          class="cursor-pointer rounded-sm whitespace-nowrap transition-all hover:bg-primary/10"
          @click="selectOption(option.value)"
        >
          <span
            class="p-1 inline-block transition-all"
            :class="modelValue === option.value ? 'font-bold' : ''"
          >
            {{ option.label }}
          </span>
        </div>
      </div>
    </Transition>
  </div>
</template>
