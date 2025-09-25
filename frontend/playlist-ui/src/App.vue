<script setup lang="ts">
import { ref, onMounted } from "vue";
import ControlsPanel from "@/components/ControlsPanel.vue";
import ResultsTable from "@/components/ResultsTable.vue";
import { fitModel, recommend, health, type RecommendationRow, API_BASE } from "@/lib/api";
import { CheckCircle2, AlertTriangle } from "lucide-vue-next";

const items = ref<RecommendationRow[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);
const fitted = ref(false);

async function runFit() {
  loading.value = true; error.value = null;
  try {
    await fitModel();
    fitted.value = true;
  } catch (e: any) {
    error.value = e.message ?? String(e);
  } finally {
    loading.value = false;
  }
}

async function runRecommend(payload: Parameters<typeof recommend>[0]) {
  loading.value = true; error.value = null;
  try {
    const res = await recommend(payload);
    items.value = res;
  } catch (e: any) {
    error.value = e.message ?? String(e);
  } finally {
    loading.value = false;
  }
}

onMounted(async () => {
  try {
    const h = await health();
    fitted.value = !!h.model_fitted;
  } catch { /* ignore */ }
});
</script>

<template>
  <div class="min-h-full">
    <header class="sticky top-0 bg-white/80 backdrop-blur border-b border-slate-100">
      <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <img src="https://fav.farm/🎵" class="w-6 h-6" alt="logo" />
          <h1 class="text-lg font-semibold text-slate-800">Playlist Generator</h1>
        </div>
        <div class="text-xs text-slate-600">
          API: <span class="font-mono">{{ API_BASE }}</span>
        </div>
      </div>
    </header>

    <main class="max-w-6xl mx-auto px-4 py-6 space-y-6">
      <div class="flex items-center gap-2 text-sm">
        <component :is="fitted ? CheckCircle2 : AlertTriangle"
                   class="w-4 h-4"
                   :class="fitted ? 'text-emerald-600' : 'text-amber-600'"/>
        <span v-if="fitted" class="text-emerald-700">Model is fitted.</span>
        <span v-else class="text-amber-700">Model not fitted yet — hit “Fit Model”.</span>
      </div>

      <ControlsPanel @fit="runFit" @recommend="runRecommend" />

      <ResultsTable :items="items" :loading="loading" :error="error" />
    </main>
  </div>
</template>
