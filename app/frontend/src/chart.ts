import { ref, computed, nextTick } from "vue";
import { fetchLog, fetchMetrics, type ParsedLogResponse } from "./service";
import { Chart, LineController, LineElement, PointElement, LinearScale, CategoryScale, Legend, Tooltip, Filler } from 'chart.js';

Chart.register(LineController, LineElement, PointElement, LinearScale, CategoryScale, Legend, Tooltip, Filler);
export type MetricsGroup = Record<string, Record<string, number>>;
export const chartCanvas = ref<HTMLCanvasElement | null>(null);
export const selectedMetrics = ref<string[]>(['recall@10', 'precision@10', 'mrr@10', 'map@10', 'hit@10', 'ndcg@10']);
export const currentLog = ref<ParsedLogResponse>({ epochs: [] });
export const showLog = ref<boolean>(false);
export let chartInstance: Chart | null = null;

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
  epochs.forEach(e => Object.keys(e.metrics).forEach(k => keys.add(k)));
  return Array.from(keys);
});

export const toggleMetric = (m: string) => {
  const i = selectedMetrics.value.indexOf(m);
  if (i >= 0) selectedMetrics.value.splice(i, 1);
  else selectedMetrics.value.push(m);
  buildChart();
};

export const toggleChart = async () => {
  showLog.value = !showLog.value;
  if (showLog.value) {
    await nextTick();
    buildChart();
  }
};

export const buildChart = () => {
  if (!chartCanvas.value) return;
  const epochs = currentLog.value.epochs;
  if (!epochs.length) return;

  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  const labels = epochs.map(e => `Epoch ${e.epoch + 1}`);
  const active = selectedMetrics.value.filter(m => metricKeys.value.includes(m));

  const datasets = active.map((metric) => {
    const colorIdx = metricKeys.value.indexOf(metric);
    const color = CHART_COLORS[colorIdx % CHART_COLORS.length];
    const data = epochs.map(e => {
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
            color: '#f1f5f9' 
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
            color: '#f1f5f9' 
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

export const currentMetricsTable = ref<{results_1: MetricsGroup; results_N: MetricsGroup; }>({ results_1: {}, results_N: {} });
export const showMetricsTable = ref<boolean>(false);
export const getMetrics = async () => {
  currentMetricsTable.value = await fetchMetrics();
};