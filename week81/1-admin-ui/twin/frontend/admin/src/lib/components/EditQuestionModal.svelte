<script lang="ts">
	import Modal from './Modal.svelte';
	import type { Question } from '$lib/types';
	import { Save, Sparkles } from 'lucide-svelte';

	let { show, question = null, onSave, onClose } = $props();

	let questionText = $state('');
	let answerText = $state('');

	$effect(() => {
		if (show) {
			questionText = question?.question ?? '';
			answerText = question?.answer ?? '';
		}
	});

	function handleSave(event: SubmitEvent) {
		event.preventDefault();
		if (questionText.trim()) {
			onSave({ question: questionText.trim(), answer: answerText.trim() || null });
		}
	}

	let title = $derived(question ? 'Edit Question' : 'Add New Question');
</script>

<Modal {show} {onClose} {title}>
	<form onsubmit={handleSave} class="space-y-8">
		<div class="space-y-6">
			<div>
				<label for="question" class="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
					Question
				</label>
				<textarea
					id="question"
					rows="4"
					bind:value={questionText}
					required
					class="block w-full rounded-xl border-0 bg-slate-50/50 py-4 px-4 text-slate-900 shadow-sm ring-1 ring-inset ring-slate-200 transition-all duration-200 placeholder:text-slate-400 focus:bg-white focus:ring-2 focus:ring-indigo-500 dark:bg-slate-800/50 dark:text-white dark:ring-slate-700 dark:focus:bg-slate-800 dark:focus:ring-indigo-400 resize-none"
					placeholder="What question should be answered in the knowledge base?"
				></textarea>
			</div>

			<div>
				<label for="answer" class="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
					Answer <span class="text-slate-500 font-normal">(optional)</span>
				</label>
				<textarea
					id="answer"
					rows="6"
					bind:value={answerText}
					class="block w-full rounded-xl border-0 bg-slate-50/50 py-4 px-4 text-slate-900 shadow-sm ring-1 ring-inset ring-slate-200 transition-all duration-200 placeholder:text-slate-400 focus:bg-white focus:ring-2 focus:ring-indigo-500 dark:bg-slate-800/50 dark:text-white dark:ring-slate-700 dark:focus:bg-slate-800 dark:focus:ring-indigo-400 resize-none"
					placeholder="Provide a comprehensive answer that will help users understand this topic..."
				></textarea>
			</div>
		</div>

		<div class="flex justify-end gap-4 border-t border-slate-200/50 pt-6 dark:border-slate-700/50">
			<button
				type="button"
				class="btn-secondary"
				onclick={onClose}
			>
				Cancel
			</button>
			<button
				type="submit"
				class="btn-primary"
			>
				<Save class="h-4 w-4" />
				{question ? 'Update Question' : 'Add Question'}
			</button>
		</div>
	</form>
</Modal>
