<script setup lang="ts">
import { reactive, computed } from "vue";
import type { RecommendRequest } from "@/lib/api";
import { Play, Settings2, SlidersHorizontal, RefreshCw } from "lucide-vue-next";

type Emits = {
  (e: "fit"): void;
  (e: "recommend", payload: RecommendRequest): void;
};

const emit = defineEmits<Emits>();

const state = reactive({
  prompt: "",
  k: 30,
  perArtistCap: 2,
  minYear: 2016 as number | null,
  minPopQuantile: 0.5,

  // advanced scoring overrides (per-request)
  showAdvanced: false,
  audioWeight: null as number | null,
  textWeight: null as number | null,
  popWeight: null as number | null,

  // future-safe fields (accepted by backend, ignored in scoring until you implement)
  featureWeightsJson: "{\n  \"danceability\": 1.0,\n  \"energy\": 1.0\n}",
  excludeFeaturesCsv: "speechiness",
});

const canRecommend = computed(() => state.prompt.trim().length > 0);

function buildPayload(): RecommendRequest {
  const payload: RecommendRequest = {
    prompt: state.prompt.trim(),
    k: state.k,
    per_artist_cap: state.perArtistCap,
    min_year: state.minYear,
    min_pop_quantile: state.minPopQuantile,
  };

  const scoring: Record<string, number> = {};
  if (state.audioWeight !== null) scoring.audio_weight = state.audioWeight;
  if (state.textWeight  !== null) scoring.text_weight  = state.textWeight;
  if (state.popWeight   !== null) scoring.pop_weight   = state.popWeight;

  const config: RecommendRequest["config"] = {};
  config.filters = {
    min_year: state.minYear,
    min_pop_quantile: state.minPopQuantile,
  };

  if (Object.keys(scoring).length > 0) {
    config.scoring = scoring as any;
  }

  // Safely parse feature weights JSON if provided
  try {
    const fw = JSON.parse(state.featureWeightsJson);
    if (fw && typeof fw === "object") {
      (config as any).audio_feature_weights = fw;
    }
  } catch {
    // ignore parse error; UI doesn't block recommend
  }

  const excludes = state.excludeFeaturesCsv
    .split(",")
    .map(s => s.trim())
    .filter(Boolean);
  if (excludes.length > 0) {
    (config as any).exclude_audio_features = excludes;
  }

  payload.config = config;
  return payload;
}

function onRecommend() {
  emit("recommend", buildPayload());
}
</script>

<template>
  <div class="bg-white rounded-2xl shadow-soft p-5">
    <div class="flex items-center gap-3 mb-4">
      <SlidersHorizontal class="w-5 h-5 text-slate-500" />
      <h2 class="text-lg font-semibold text-slate-800">Generate Recommendations</h2>
    </div>

    <div class="grid md:grid-cols-2 gap-4">
      <div class="md:col-span-2">
        <label class="block text-sm font-medium text-slate-700 mb-1">Prompt</label>
        <textarea
          v-model="state.prompt"
          rows="2"
          placeholder="e.g., melodic deep house for studying"
          class="w-full rounded-xl border border-slate-200 focus:ring-2 focus:ring-indigo-500 focus:border-transparent px-3 py-2"
        />
      </div>

      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1">Top K</label>
        <input type="number" v-model.number="state.k" min="1" max="200"
               class="w-full rounded-xl border border-slate-200 px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"/>
      </div>

      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1">Per-Artist Cap</label>
        <input type="number" v-model.number="state.perArtistCap" min="1" max="10"
               class="w-full rounded-xl border border-slate-200 px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"/>
      </div>

      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1">Min Year</label>
        <input type="number" v-model.number="state.minYear" placeholder="e.g., 2016"
               class="w-full rounded-xl border border-slate-200 px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"/>
        <p class="text-xs text-slate-500 mt-1">Use blank to disable year filtering.</p>
      </div>

      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1">
          Min Popularity Quantile <span class="text-slate-400">(0–0.99)</span>
        </label>
        <input type="number" v-model.number="state.minPopQuantile" min="0" max="0.99" step="0.01"
               class="w-full rounded-xl border border-slate-200 px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"/>
      </div>
    </div>

    <div class="mt-4 flex items-center justify-between">
      <button
        class="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-600 text-white hover:bg-emerald-700"
        @click="$emit('fit')"
        type="button"
      >
        <RefreshCw class="w-4 h-4" /> Fit Model
      </button>

      <div class="flex items-center gap-3">
        <button
          class="inline-flex items-center gap-2 px-3 py-2 rounded-xl border text-slate-700 border-slate-200 hover:bg-slate-50"
          type="button"
          @click="state.showAdvanced = !state.showAdvanced"
        >
          <Settings2 class="w-4 h-4" /> Advanced
        </button>
        <button
          class="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50"
          :disabled="!canRecommend"
          @click="onRecommend"
          type="button"
        >
          <Play class="w-4 h-4" /> Recommend
        </button>
      </div>
    </div>

    <transition name="fade">
      <div v-if="state.showAdvanced" class="mt-5 grid md:grid-cols-3 gap-4">
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">Audio Weight</label>
          <input type="number" step="0.05" v-model.number="state.audioWeight" placeholder="e.g., 0.5"
                 class="w-full rounded-xl border border-slate-200 px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"/>
        </div>
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">Text Weight</label>
          <input type="number" step="0.05" v-model.number="state.textWeight" placeholder="e.g., 0.4"
                 class="w-full rounded-xl border border-slate-200 px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"/>
        </div>
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">Popularity Weight</label>
          <input type="number" step="0.05" v-model.number="state.popWeight" placeholder="e.g., 0.1"
                 class="w-full rounded-xl border border-slate-200 px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"/>
        </div>

        <div class="md:col-span-2">
          <label class="block text-sm font-medium text-slate-700 mb-1">Audio Feature Weights (JSON)</label>
          <textarea v-model="state.featureWeightsJson" rows="4"
                    class="w-full rounded-xl border border-slate-200 px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"></textarea>
          <p class="text-xs text-slate-500 mt-1">Accepted by API (not applied until you implement per-feature weighting).</p>
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">Exclude Audio Features (CSV)</label>
          <input type="text" v-model="state.excludeFeaturesCsv" placeholder="speechiness,liveness"
                 class="w-full rounded-xl border border-slate-200 px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"/>
        </div>
      </div>
    </transition>
  </div>
</template>

<style scoped>
.fade-enter-active,.fade-leave-active{ transition: opacity .15s ease }
.fade-enter-from,.fade-leave-to{ opacity: 0 }
</style>
