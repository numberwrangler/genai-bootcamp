<script lang="ts">
	import { X } from 'lucide-svelte';
	import { quintOut } from 'svelte/easing';
	import { fly, fade, scale } from 'svelte/transition';

	let { show, onClose, title, children } = $props();

	let dialog = $state<HTMLDivElement | undefined>();

	$effect(() => {
		if (show) {
			dialog?.focus();
		}
	});
</script>

{#if show}
	<div
		transition:fade={{ duration: 200 }}
		role="dialog"
		aria-modal="true"
		tabindex="-1"
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
		onclick={(e) => e.target === e.currentTarget && onClose()}
		onkeydown={(e) => e.key === 'Escape' && onClose()}
	>
		<div
			bind:this={dialog}
			transition:scale={{ duration: 300, easing: quintOut, start: 0.95 }}
			class="w-full max-w-2xl glass-card glass-card-dark rounded-2xl shadow-2xl shadow-slate-900/25 overflow-hidden"
		>
			<div class="flex items-center justify-between border-b border-slate-200/50 bg-gradient-to-r from-slate-50/50 to-white/50 px-8 py-6 dark:border-slate-700/50 dark:from-slate-800/50 dark:to-slate-900/50">
				<h2 class="text-xl font-bold text-slate-900 dark:text-slate-50">{title}</h2>
				<button
					onclick={onClose}
					class="rounded-xl p-2 text-slate-400 transition-all duration-200 hover:bg-slate-100 hover:text-slate-600 hover:scale-110 dark:hover:bg-slate-800 dark:hover:text-slate-300"
				>
					<X class="h-5 w-5" />
					<span class="sr-only">Close</span>
				</button>
			</div>
			<div class="p-8 text-slate-700 dark:text-slate-300">
				{@render children()}
			</div>
		</div>
	</div>
{/if}
