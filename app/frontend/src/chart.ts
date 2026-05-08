import { ref, computed, nextTick } from "vue";
import { fetchLog, fetchMetrics } from "./service";
import { Chart, LineController, LineElement, PointElement, LinearScale, CategoryScale, Legend, Tooltip, Filler } from 'chart.js';

Chart.register(LineController, LineElement, PointElement, LinearScale, CategoryScale, Legend, Tooltip, Filler);
export const chartCanvas = ref<HTMLCanvasElement | null>(null);
export const selectedMetrics = ref<string[]>(['recall@10', 'precision@10', 'mrr@10', 'map@10', 'hit@10', 'ndcg@10']);
export const currentLog = ref<TrainingLog>({ epochs: [] });
export const showLog = ref<boolean>(false);
export const currentMetricsTable = ref<Metrics>();
export const showMetricsTable = ref<boolean>(false);
export let chartInstance: Chart | null = null;

export const getMetrics = async () => {
  currentMetricsTable.value = await fetchMetrics();
};

export const loadLog = async () => {
  currentLog.value = await fetchLog();
};

export const CHART_COLORS = [
  '#6366f1', '#f43f5e', '#10b981', '#f59e0b', '#3b82f6',
  '#8b5cf6', '#ec4899', '#14b8a6', '#ef4444', '#84cc16',
  '#06b6d4', '#d946ef', '#f97316', '#22d3ee', '#a3e635',
];

export const metricKeys = computed(() => {
  const epochs = currentLog.value.epochs;
  if (!epochs.length) return [];
  const keys = new Set<string>();
  keys.add('train_loss');
  keys.add('valid_score');
  epochs.forEach((e: Epoch) => Object.keys(e.metrics).forEach((k: string) => keys.add(k)));
  const arr = Array.from(keys);
  arr.sort((a, b) => {
    if (a === 'train_loss') return -1;
    if (b === 'train_loss') return 1;
    if (a === 'valid_score') return -1;
    if (b === 'valid_score') return 1;
    
    const [nameA, kA] = a.split('@');
    const [nameB, kB] = b.split('@');
    
    if (nameA === nameB && kA !== undefined && kB !== undefined) {
      return parseInt(kA) - parseInt(kB);
    }
    
    return a.localeCompare(b);
  });
  return arr;
});

export const toggleMetric = (metric: string) => {
  const i = selectedMetrics.value.indexOf(metric);
  if (i >= 0) selectedMetrics.value.splice(i, 1);
  else selectedMetrics.value.push(metric);
  buildChart();
};

export const toggleChart = async () => {
  showMetricsTable.value = false;
  showLog.value = !showLog.value;
  if (showLog.value) {
    await nextTick();
    buildChart();
  }
};

export const toggleMetricsTable = () => {
  showLog.value = false;
  showMetricsTable.value = !showMetricsTable.value;
};

export const buildChart = () => {
  if (!chartCanvas.value) return;
  const epochs = currentLog.value.epochs;
  if (!epochs.length) return;

  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  const labels = epochs.map((e: Epoch) => `Epoch ${e.epoch + 1}`);
  const active = selectedMetrics.value.filter((m: string) => metricKeys.value.includes(m));

  const datasets = active.map((metric: string) => {
    const colorIdx = metricKeys.value.indexOf(metric);
    const color = CHART_COLORS[colorIdx % CHART_COLORS.length];
    const data = epochs.map((e: Epoch) => {
      if (metric === 'train_loss') return e.train_loss ?? NaN;
      if (metric === 'valid_score') return e.valid_score ?? NaN;
      return e.metrics[metric] ?? NaN;
    });
    const isLoss = metric === 'train_loss';
    return {
      label: metric,
      data,
      borderColor: color,
      backgroundColor: color + '22',
      pointBackgroundColor: color,
      pointRadius: 1,
      pointHoverRadius: 4,
      borderWidth: 2,
      tension: 0.3,
      yAxisID: isLoss ? 'yLoss' : 'y',
    };
  });

  const hasLoss = active.includes('train_loss');

  chartInstance = new Chart(chartCanvas.value, {
    type: 'line',
    data: { 
      labels, 
      datasets 
    },
    options: {
      responsive: true,
      layout: {
        padding: 10,
      },
      maintainAspectRatio: false,
      interaction: { 
        mode: 'index', 
        intersect: false 
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { 
            boxWidth: 12, 
            padding: 20, 
            font: { size: 15 }, 
            usePointStyle: true, 
            pointStyle: 'circle', 
            boxHeight: 10 
          },
        },
        tooltip: {
          backgroundColor: '#1e293b',
          titleFont: { size: 15 },
          bodyFont: { size: 15 },
          padding: 5,
        },
      },
      scales: {
        x: {
          grid: { 
            color: '#cbd5e1' 
          },
          ticks: {
            font: { size: 12 },
            maxRotation: 45,
            autoSkip: false,
            callback: function (value, index) {
              return (index + 1) % 5 === 0 || index === 0 
              ? this.getLabelForValue(value as number) 
              : '';
            }
          },
        },
        y: {
          position: 'left',
          beginAtZero: true,
          grid: { 
            color: '#cbd5e1' 
          },
          ticks: { 
            font: { size: 12 }, 
            stepSize: 0.005 
          },
          title: { 
            display: true, 
            text: 'Metrics', 
            font: { size: 15 } 
          },
        },
        ...(hasLoss ? {
          yLoss: {
            position: 'right' as const,
            grid: { 
                drawOnChartArea: false 
            },
            ticks: { 
                font: { size: 9 } 
            },
            title: { 
                display: true, 
                text: 'Train Loss', 
                font: { size: 15 } 
            },
          },
        } : {}),
      },
    },
  });
};