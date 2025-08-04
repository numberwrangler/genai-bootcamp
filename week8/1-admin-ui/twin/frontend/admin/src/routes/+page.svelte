<script lang="ts">
	import { onMount } from 'svelte';
	import {
		listQuestions,
		addQuestion,
		updateQuestion,
		deleteQuestion,
		syncToKnowledgeBase
	} from '$lib/api';
	import type { Question, SyncResponse } from '$lib/types';
	import EditQuestionModal from '$lib/components/EditQuestionModal.svelte';
	import DeleteConfirmModal from '$lib/components/DeleteConfirmModal.svelte';
	import {
		Plus,
		RefreshCw,
		Trash2,
		Edit,
		CheckCircle,
		XCircle,
		AlertCircle,
		Database,
		X,
		Sparkles,
		Brain,
		TrendingUp,
		Filter
	} from 'lucide-svelte';

	let questions: Question[] = [];
	let isLoading = true;
	let error: string | null = null;
	let unansweredOnly = false;
	let showEditModal = false;
	let showDeleteModal = false;
	let selectedQuestion: Question | null = null;

	let showSyncNotification = false;
	let syncResult: SyncResponse | null = null;
	let syncError: string | null = null;
	let isSyncing = false;

	async function loadQuestions() {
		isLoading = true;
		error = null;
		try {
			questions = await listQuestions(unansweredOnly);
		} catch (e: any) {
			error = e.message || 'Failed to load questions.';
		} finally {
			isLoading = false;
		}
	}

	$: unansweredOnly, loadQuestions();

	function handleAddClick() {
		selectedQuestion = null;
		showEditModal = true;
	}

	function handleEditClick(question: Question) {
		selectedQuestion = question;
		showEditModal = true;
	}

	function handleDeleteClick(question: Question) {
		selectedQuestion = question;
		showDeleteModal = true;
	}

	async function handleSaveQuestion(data: { question: string; answer: string | null }) {
		try {
			if (selectedQuestion) {
				await updateQuestion(selectedQuestion.question_id, data);
			} else {
				await addQuestion(data.question, data.answer);
			}
			showEditModal = false;
			await loadQuestions();
		} catch (e: any) {
			error = e.message || 'Failed to save question.';
		}
	}

	async function confirmDelete() {
		if (selectedQuestion) {
			try {
				await deleteQuestion(selectedQuestion.question_id);
				showDeleteModal = false;
				await loadQuestions();
			} catch (e: any) {
				error = e.message || 'Failed to delete question.';
			}
		}
	}

	async function handleSync() {
		isSyncing = true;
		syncError = null;
		syncResult = null;
		try {
			syncResult = await syncToKnowledgeBase();
			showSyncNotification = true;
			setTimeout(() => (showSyncNotification = false), 5000);
			await loadQuestions();
		} catch (e: any) {
			syncError = e.message || 'Failed to sync.';
			showSyncNotification = true;
			setTimeout(() => (showSyncNotification = false), 5000);
		} finally {
			isSyncing = false;
		}
	}

	$: totalQuestions = questions.length;
	$: answeredQuestions = questions.filter(q => q.answer).length;
	$: syncedQuestions = questions.filter(q => q.processed).length;
</script>

<div class="min-h-screen">
	<!-- Hero Section -->
	<div class="relative overflow-hidden bg-gradient-to-br from-indigo-50 via-white to-purple-50 dark:from-slate-900 dark:via-slate-800 dark:to-indigo-900">
		<div class="absolute inset-0 overflow-hidden">
			<div class="absolute -top-40 -right-40 h-80 w-80 rounded-full bg-gradient-to-br from-indigo-400/10 to-purple-600/10 blur-3xl animate-float"></div>
			<div class="absolute -bottom-40 -left-40 h-80 w-80 rounded-full bg-gradient-to-br from-blue-400/10 to-indigo-600/10 blur-3xl animate-float" style="animation-delay: -3s;"></div>
		</div>
		
		<div class="relative mx-auto max-w-7xl px-6 py-16 lg:px-8">
			<div class="text-center mb-12">
				<h1 class="text-5xl font-bold tracking-tight mb-4">
					<span class="bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
						Knowledge Base
					</span>
					<br />
					<span class="text-slate-900 dark:text-white">Management Hub</span>
				</h1>
				<p class="text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
					Curate, manage, and synchronize your AI knowledge base with enterprise-grade tools
				</p>
			</div>

			<!-- Stats Cards -->
			<div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
				<div class="glass-card glass-card-dark rounded-2xl p-6 text-center group hover:scale-105 transition-all duration-300">
					<div class="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 shadow-lg shadow-blue-500/25">
						<Database class="h-6 w-6 text-white" />
					</div>
					<div class="text-3xl font-bold text-slate-900 dark:text-white mb-1">{totalQuestions}</div>
					<div class="text-sm font-medium text-slate-600 dark:text-slate-400">Total Questions</div>
				</div>
				
				<div class="glass-card glass-card-dark rounded-2xl p-6 text-center group hover:scale-105 transition-all duration-300">
					<div class="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-green-600 shadow-lg shadow-emerald-500/25">
						<CheckCircle class="h-6 w-6 text-white" />
					</div>
					<div class="text-3xl font-bold text-slate-900 dark:text-white mb-1">{answeredQuestions}</div>
					<div class="text-sm font-medium text-slate-600 dark:text-slate-400">Answered</div>
				</div>
				
				<div class="glass-card glass-card-dark rounded-2xl p-6 text-center group hover:scale-105 transition-all duration-300">
					<div class="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 shadow-lg shadow-purple-500/25">
						<Brain class="h-6 w-6 text-white" />
					</div>
					<div class="text-3xl font-bold text-slate-900 dark:text-white mb-1">{syncedQuestions}</div>
					<div class="text-sm font-medium text-slate-600 dark:text-slate-400">Synced to KB</div>
				</div>
			</div>

			<!-- Action Bar -->
			<div class="flex flex-wrap items-center justify-between gap-4 mb-8">
				<div class="flex items-center gap-4">
					<div class="flex items-center gap-3 glass-card glass-card-dark rounded-xl px-4 py-3">
						<Filter class="h-5 w-5 text-slate-500" />
						<input
							id="unansweredOnly"
							type="checkbox"
							bind:checked={unansweredOnly}
							class="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 dark:border-slate-600 dark:bg-slate-700"
						/>
						<label for="unansweredOnly" class="text-sm font-medium text-slate-700 dark:text-slate-300">
							Show unanswered only
						</label>
					</div>
				</div>
				
				<div class="flex gap-3">
					<button
						type="button"
						onclick={handleSync}
						disabled={isSyncing}
						class="btn-secondary disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100"
					>
						<RefreshCw class="h-5 w-5 {isSyncing ? 'animate-spin' : ''}" />
						{isSyncing ? 'Syncing...' : 'Sync to KB'}
					</button>
					<button
						type="button"
						onclick={handleAddClick}
						class="btn-primary"
					>
						<Plus class="h-5 w-5" />
						Add Question
					</button>
				</div>
			</div>
		</div>
	</div>

	<!-- Content Section -->
	<div class="mx-auto max-w-7xl px-6 py-12 lg:px-8">
		{#if isLoading}
			<div class="flex flex-col items-center justify-center py-20">
				<div class="relative">
					<div class="h-16 w-16 animate-spin rounded-full border-4 border-indigo-200 border-t-indigo-600"></div>
					<div class="absolute inset-0 h-16 w-16 animate-pulse rounded-full bg-indigo-100/50"></div>
				</div>
				<p class="mt-4 text-lg font-medium text-slate-600 dark:text-slate-400">Loading questions...</p>
			</div>
		{:else if error}
			<div class="glass-card glass-card-dark rounded-2xl p-8">
				<div class="flex items-start gap-4">
					<div class="flex-shrink-0">
						<div class="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-red-500 to-pink-600 shadow-lg shadow-red-500/25">
							<AlertCircle class="h-6 w-6 text-white" />
						</div>
					</div>
					<div class="flex-1">
						<h3 class="text-lg font-semibold text-slate-900 dark:text-white mb-2">
							Something went wrong
						</h3>
						<p class="text-slate-600 dark:text-slate-400">{error}</p>
						<button
							onclick={() => loadQuestions()}
							class="mt-4 btn-primary"
						>
							<RefreshCw class="h-4 w-4" />
							Try Again
						</button>
					</div>
				</div>
			</div>
		{:else if questions.length === 0}
			<div class="text-center py-20">
				<div class="mx-auto mb-8 flex h-24 w-24 items-center justify-center rounded-2xl bg-gradient-to-br from-slate-100 to-slate-200 dark:from-slate-800 dark:to-slate-700">
					<Database class="h-12 w-12 text-slate-400" />
				</div>
				<h3 class="text-2xl font-bold text-slate-900 dark:text-white mb-2">No questions yet</h3>
				<p class="text-lg text-slate-600 dark:text-slate-400 mb-8 max-w-md mx-auto">
					Start building your knowledge base by adding your first question and answer pair.
				</p>
				<button
					type="button"
					onclick={handleAddClick}
					class="btn-primary text-lg px-8 py-4"
				>
					<Plus class="h-6 w-6" />
					Create First Question
				</button>
			</div>
		{:else}
			<div class="grid gap-6">
				{#each questions as question (question.question_id)}
					<div class="glass-card glass-card-dark rounded-2xl overflow-hidden group hover:shadow-2xl hover:shadow-slate-900/10 transition-all duration-300 hover:scale-[1.02]">
						<div class="p-8">
							<div class="flex items-start justify-between gap-6">
								<div class="flex-1 space-y-4">
									<div class="flex items-start gap-4">
										<div class="flex-shrink-0 mt-1">
											<div class="h-3 w-3 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500"></div>
										</div>
										<div class="flex-1">
											<h3 class="text-lg font-semibold text-slate-900 dark:text-white leading-relaxed">
												{question.question}
											</h3>
										</div>
									</div>
									<div class="ml-7">
										<p class="text-slate-600 dark:text-slate-400 leading-relaxed">
											{question.answer ?? 'No answer provided yet.'}
										</p>
									</div>
								</div>
								<div class="flex-shrink-0">
									{#if question.processed}
										<span class="status-badge-synced">
											<CheckCircle class="h-3 w-3" />
											Synced
										</span>
									{:else}
										<span class="status-badge-pending">
											<RefreshCw class="h-3 w-3" />
											Pending
										</span>
									{/if}
								</div>
							</div>
						</div>
						<div class="border-t border-slate-200/50 bg-slate-50/50 px-8 py-4 dark:border-slate-700/50 dark:bg-slate-800/50">
							<div class="flex items-center justify-end gap-2">
								<button
									onclick={() => handleEditClick(question)}
									class="inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium text-slate-700 transition-all duration-200 hover:bg-slate-100 hover:scale-105 dark:text-slate-300 dark:hover:bg-slate-700"
								>
									<Edit class="h-4 w-4" />
									Edit
								</button>
								<button
									onclick={() => handleDeleteClick(question)}
									class="inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium text-red-600 transition-all duration-200 hover:bg-red-50 hover:scale-105 dark:text-red-400 dark:hover:bg-red-900/20"
								>
									<Trash2 class="h-4 w-4" />
									Delete
								</button>
							</div>
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Sync Notification -->
	{#if showSyncNotification}
		<div class="fixed bottom-8 right-8 z-50 w-full max-w-sm">
			<div class="glass-card glass-card-dark rounded-2xl p-6 shadow-2xl {syncError ? 'ring-2 ring-red-500/50' : 'ring-2 ring-emerald-500/50'}">
				<div class="flex items-start gap-4">
					<div class="flex-shrink-0">
						{#if syncError}
							<div class="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-red-500 to-pink-600">
								<XCircle class="h-5 w-5 text-white" />
							</div>
						{:else}
							<div class="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-green-600">
								<CheckCircle class="h-5 w-5 text-white" />
							</div>
						{/if}
					</div>
					<div class="flex-1">
						<p class="font-semibold text-slate-900 dark:text-white">
							{#if syncError}
								Sync Failed
							{:else if syncResult}
								Sync Successful
							{/if}
						</p>
						<p class="text-sm text-slate-600 dark:text-slate-400 mt-1">
							{#if syncError}
								{syncError}
							{:else if syncResult}
								{syncResult.status}
							{/if}
						</p>
					</div>
					<button
						onclick={() => (showSyncNotification = false)}
						class="flex-shrink-0 rounded-lg p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
					>
						<X class="h-5 w-5" />
					</button>
				</div>
			</div>
		</div>
	{/if}
</div>

<EditQuestionModal
	show={showEditModal}
	question={selectedQuestion}
	onSave={handleSaveQuestion}
	onClose={() => (showEditModal = false)}
/>

<DeleteConfirmModal
	show={showDeleteModal}
	questionText={selectedQuestion?.question ?? ''}
	onConfirm={confirmDelete}
	onCancel={() => (showDeleteModal = false)}
/>
