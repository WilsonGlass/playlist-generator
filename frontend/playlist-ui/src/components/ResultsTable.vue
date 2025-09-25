<script setup lang="ts">
import type { RecommendationRow } from "@/lib/api";

defineProps<{
  items: RecommendationRow[];
  loading?: boolean;
  error?: string | null;
}>();
</script>

<template>
  <div class="bg-white rounded-2xl shadow-soft p-5">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-lg font-semibold text-slate-800">Results</h2>
      <span v-if="loading" class="text-sm text-slate-500">Loading…</span>
    </div>

    <p v-if="error" class="text-sm text-red-600 mb-3">{{ error }}</p>
    <p v-else-if="!loading && items.length === 0" class="text-sm text-slate-500">No results yet.</p>

    <div v-if="items.length" class="overflow-x-auto">
      <table class="min-w-full text-sm">
        <thead class="text-left bg-slate-50">
          <tr class="text-slate-600">
            <th class="py-2 px-3">Track</th>
            <th class="py-2 px-3">Artist</th>
            <th class="py-2 px-3">Genre</th>
            <th class="py-2 px-3">Subgenre</th>
            <th class="py-2 px-3">Release</th>
            <th class="py-2 px-3">Pop</th>
            <th class="py-2 px-3">Score</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in items" :key="r.track_id" class="border-t border-slate-100 hover:bg-slate-50">
            <td class="py-2 px-3 font-medium text-slate-800">{{ r.track_name }}</td>
            <td class="py-2 px-3 text-slate-700">{{ r.track_artist }}</td>
            <td class="py-2 px-3 text-slate-600">{{ r.playlist_genre ?? '—' }}</td>
            <td class="py-2 px-3 text-slate-600">{{ r.playlist_subgenre ?? '—' }}</td>
            <td class="py-2 px-3 text-slate-600">{{ r.track_album_release_date ?? '—' }}</td>
            <td class="py-2 px-3 text-slate-600">{{ r.track_popularity ?? '—' }}</td>
            <td class="py-2 px-3 text-slate-800">{{ r.score.toFixed(4) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
