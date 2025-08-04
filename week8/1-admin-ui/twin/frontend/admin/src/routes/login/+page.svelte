<script lang="ts">
	import { goto } from '$app/navigation';
	import { startPasswordlessSignIn, completePasswordlessSignIn } from '$lib/auth';
	import { Mail, KeyRound, AlertTriangle, Sparkles, Shield } from 'lucide-svelte';

	let email = '';
	let code = '';
	let emailSent = false;
	let loading = false;
	let error: string | null = null;

	async function handleEmailSubmit(event: SubmitEvent) {
		event.preventDefault();
		error = null;
		loading = true;
		try {
			const success = await startPasswordlessSignIn(email);
			if (success) {
				emailSent = true;
			} else {
				error = 'Could not start sign-in process. Please try again.';
			}
		} catch (e: any) {
			error = e.message || 'An unknown error occurred.';
		} finally {
			loading = false;
		}
	}

	async function handleCodeSubmit(event: SubmitEvent) {
		event.preventDefault();
		error = null;
		loading = true;
		try {
			const success = await completePasswordlessSignIn(code);
			if (success) {
				await goto('/admin/');
			} else {
				error = 'Sign in failed. Invalid code.';
			}
		} catch (e: any) {
			error = e.message || 'An unknown error occurred.';
		} finally {
			loading = false;
		}
	}
</script>

<div class="flex min-h-screen flex-col items-center justify-center p-6 relative overflow-hidden">
	<!-- Background decorations -->
	<div class="absolute inset-0 overflow-hidden">
		<div class="absolute -top-40 -right-40 h-80 w-80 rounded-full bg-gradient-to-br from-indigo-400/20 to-purple-600/20 blur-3xl animate-float"></div>
		<div class="absolute -bottom-40 -left-40 h-80 w-80 rounded-full bg-gradient-to-br from-blue-400/20 to-indigo-600/20 blur-3xl animate-float" style="animation-delay: -3s;"></div>
	</div>

	<div class="relative z-10 w-full max-w-md space-y-8">
		<div class="text-center">
			<div class="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-2xl shadow-indigo-500/25">
				<Shield class="h-8 w-8 text-white" />
			</div>
			<h1 class="text-4xl font-bold tracking-tight">
				<span class="bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
					Admin Portal
				</span>
			</h1>
			<p class="mt-3 text-lg text-slate-600 dark:text-slate-400">
				{#if emailSent}
					<span class="flex items-center justify-center gap-2">
						<Sparkles class="h-5 w-5 text-indigo-500" />
						Code sent to <strong class="font-semibold text-slate-800 dark:text-slate-200">{email}</strong>
					</span>
				{:else}
					Secure passwordless authentication
				{/if}
			</p>
		</div>

		<div class="glass-card glass-card-dark rounded-2xl p-8 shadow-2xl shadow-slate-900/10">
			{#if error}
				<div class="mb-6 flex items-start space-x-3 rounded-xl border border-red-200 bg-gradient-to-r from-red-50 to-pink-50 p-4 dark:border-red-800/50 dark:from-red-900/20 dark:to-pink-900/20">
					<AlertTriangle class="h-5 w-5 flex-shrink-0 text-red-500" />
					<div class="flex-1">
						<h3 class="font-semibold text-red-800 dark:text-red-200">Authentication Error</h3>
						<p class="text-sm text-red-700 dark:text-red-300">{error}</p>
					</div>
				</div>
			{/if}

			{#if !emailSent}
				<form class="space-y-6" onsubmit={handleEmailSubmit}>
					<div>
						<label for="email" class="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
							Email address
						</label>
						<div class="relative">
							<Mail class="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
							<input
								id="email"
								name="email"
								type="email"
								autocomplete="email"
								required
								bind:value={email}
								class="block w-full rounded-xl border-0 bg-slate-50/50 py-4 pl-12 pr-4 text-slate-900 shadow-sm ring-1 ring-inset ring-slate-200 transition-all duration-200 placeholder:text-slate-400 focus:bg-white focus:ring-2 focus:ring-indigo-500 dark:bg-slate-800/50 dark:text-white dark:ring-slate-700 dark:focus:bg-slate-800 dark:focus:ring-indigo-400"
								placeholder="admin@company.com"
							/>
						</div>
					</div>

					<button
						type="submit"
						disabled={loading || !email}
						class="btn-primary w-full py-4 text-base font-semibold disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100 disabled:hover:shadow-lg"
					>
						{#if loading}
							<div class="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
							Sending...
						{:else}
							<Mail class="h-5 w-5" />
							Send Authentication Code
						{/if}
					</button>
				</form>
			{:else}
				<form class="space-y-6" onsubmit={handleCodeSubmit}>
					<div>
						<label for="code" class="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
							Verification code
						</label>
						<div class="relative">
							<KeyRound class="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
							<input
								id="code"
								name="code"
								type="text"
								required
								bind:value={code}
								class="block w-full rounded-xl border-0 bg-slate-50/50 py-4 pl-12 pr-4 text-slate-900 shadow-sm ring-1 ring-inset ring-slate-200 transition-all duration-200 placeholder:text-slate-400 focus:bg-white focus:ring-2 focus:ring-indigo-500 dark:bg-slate-800/50 dark:text-white dark:ring-slate-700 dark:focus:bg-slate-800 dark:focus:ring-indigo-400"
								placeholder="Enter 8-digit code"
								maxlength="8"
							/>
						</div>
					</div>

					<div class="space-y-4">
						<button
							type="submit"
							disabled={loading || !code}
							class="btn-primary w-full py-4 text-base font-semibold disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100 disabled:hover:shadow-lg"
						>
							{#if loading}
								<div class="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
								Verifying...
							{:else}
								<Shield class="h-5 w-5" />
								Access Dashboard
							{/if}
						</button>
						<button
							type="button"
							onclick={() => {
								emailSent = false;
								error = null;
								code = '';
							}}
							class="w-full rounded-xl py-3 text-center text-sm font-semibold text-indigo-600 transition-colors hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300"
						>
							← Use different email address
						</button>
					</div>
				</form>
			{/if}
		</div>

		<div class="text-center">
			<p class="text-sm text-slate-500 dark:text-slate-400">
				Secured with enterprise-grade authentication
			</p>
		</div>
	</div>
</div>
