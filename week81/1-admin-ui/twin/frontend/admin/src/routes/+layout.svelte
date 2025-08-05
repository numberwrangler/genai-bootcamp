<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';
	import { signOutUser } from '$lib/auth';
	import { goto } from '$app/navigation';
	import { LogOut, Sparkles } from 'lucide-svelte';

	let { data, children } = $props();

	async function handleSignOut() {
		await signOutUser();
		await goto('/admin/login');
	}
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>Admin Dashboard</title>
	<link rel="preconnect" href="https://fonts.googleapis.com">
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
	<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 font-sans text-slate-900 antialiased dark:from-slate-950 dark:via-slate-900 dark:to-indigo-950 dark:text-slate-50" style="font-family: 'Inter', sans-serif;">
	{#if data.isAuthenticated}
		<header class="sticky top-0 z-50 w-full border-b border-white/20 bg-white/80 backdrop-blur-xl shadow-lg shadow-slate-900/5 dark:border-slate-700/50 dark:bg-slate-900/80 dark:shadow-2xl dark:shadow-slate-900/20">
			<div class="mx-auto flex h-20 max-w-7xl items-center justify-between px-6 lg:px-8">
				<a href="/admin" class="flex items-center gap-3 group">
					<div class="relative">
						<img class="h-10 w-auto transition-transform duration-300 group-hover:scale-110" src={favicon} alt="Logo" />
						<div class="absolute -inset-2 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 opacity-0 blur transition-opacity duration-300 group-hover:opacity-20"></div>
					</div>
					<div class="flex flex-col">
						<span class="text-xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">Admin Panel</span>
						<span class="text-xs text-slate-500 dark:text-slate-400">Knowledge Base Management</span>
					</div>
				</a>
				<div class="flex items-center gap-4">
					<button
						onclick={handleSignOut}
						type="button"
						class="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-100/80 px-4 py-2.5 text-sm font-semibold text-slate-700 backdrop-blur-sm transition-all duration-200 hover:bg-slate-200/80 hover:scale-105 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 dark:bg-slate-800/80 dark:text-slate-200 dark:hover:bg-slate-700/80"
					>
						<LogOut class="h-4 w-4" aria-hidden="true" />
						Sign out
					</button>
				</div>
			</div>
		</header>
	{/if}

	<main class="relative">
		{@render children()}
	</main>
</div>
