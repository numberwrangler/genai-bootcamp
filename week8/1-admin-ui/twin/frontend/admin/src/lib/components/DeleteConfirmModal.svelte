<script lang="ts">
	import Modal from './Modal.svelte';
	import { Trash2, AlertTriangle } from 'lucide-svelte';

	let { show, onConfirm, onCancel, questionText } = $props();
</script>

<Modal {show} onClose={onCancel} title="Confirm Deletion">
	<div class="space-y-8">
		<div class="flex items-start gap-6">
			<div class="flex-shrink-0">
				<div class="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-red-500 to-pink-600 shadow-lg shadow-red-500/25">
					<AlertTriangle class="h-8 w-8 text-white" />
				</div>
			</div>
			<div class="flex-1">
				<h3 class="text-lg font-bold text-slate-900 dark:text-white mb-3">
					Delete this question?
				</h3>
				<p class="text-slate-600 dark:text-slate-400 mb-4">
					This action cannot be undone. The question and its answer will be permanently removed from your knowledge base.
				</p>
				<div class="rounded-xl bg-slate-50 p-4 dark:bg-slate-800/50">
					<p class="text-sm font-medium text-slate-700 dark:text-slate-300 italic">
						"{questionText}"
					</p>
				</div>
			</div>
		</div>

		<div class="flex justify-end gap-4 border-t border-slate-200/50 pt-6 dark:border-slate-700/50">
			<button
				type="button"
				class="btn-secondary"
				onclick={onCancel}
			>
				Keep Question
			</button>
			<button
				type="button"
				class="inline-flex items-center gap-x-2 rounded-xl bg-gradient-to-r from-red-600 to-pink-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-red-500/25 transition-all duration-200 hover:shadow-xl hover:shadow-red-500/40 hover:scale-105"
				onclick={onConfirm}
			>
				<Trash2 class="h-4 w-4" />
				Delete Forever
			</button>
		</div>
	</div>
</Modal>
