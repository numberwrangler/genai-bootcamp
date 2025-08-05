import { fetchAuthSession } from '@aws-amplify/auth';
import type { Question, UpdateQuestionPayload, SyncResponse } from '$lib/types';

async function getHeaders() {
	try {
		const session = await fetchAuthSession();
		const token = session.tokens?.idToken?.toString();
		if (!token) {
			throw new Error('No ID token found in session');
		}
		return {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		};
	} catch (error) {
		console.error('Error getting auth session:', error);
		throw new Error('Not authenticated');
	}
}

export async function listQuestions(unansweredOnly = false): Promise<Question[]> {
	const headers = await getHeaders();
	let url = '/adminapi/questions';
	if (unansweredOnly) {
		url += '?unansweredOnly=true';
	}

	const response = await fetch(url, {
		method: 'GET',
		headers
	});

	if (!response.ok) {
		const errorBody = await response.text();
		console.error('Failed to list questions:', errorBody);
		throw new Error(`Failed to list questions: ${response.statusText}`);
	}

	return await response.json();
}

export async function syncToKnowledgeBase(): Promise<SyncResponse> {
	const headers = await getHeaders();
	const url = '/adminapi/sync';

	const response = await fetch(url, {
		method: 'POST',
		headers
	});

	if (!response.ok) {
		const errorBody = await response.text();
		console.error('Failed to sync to knowledge base:', errorBody);
		throw new Error(`Failed to sync to knowledge base: ${response.statusText}`);
	}

	return await response.json();
}

export async function deleteQuestion(question_id: string): Promise<void> {
	const headers = await getHeaders();
	const url = `/adminapi/questions/${question_id}`;

	const response = await fetch(url, {
		method: 'DELETE',
		headers
	});

	if (!response.ok) {
		const errorBody = await response.text();
		console.error('Failed to delete question:', errorBody);
		throw new Error(`Failed to delete question: ${response.statusText}`);
	}
}

export async function updateQuestion(
	question_id: string,
	updates: UpdateQuestionPayload
): Promise<Question> {
	const headers = await getHeaders();
	const url = `/adminapi/questions/${question_id}/update`;

	const response = await fetch(url, {
		method: 'POST',
		headers,
		body: JSON.stringify(updates)
	});

	if (!response.ok) {
		const errorBody = await response.text();
		console.error('Failed to update question:', errorBody);
		throw new Error(`Failed to update question: ${response.statusText}`);
	}

	return await response.json();
}

export async function answerQuestion(
	question_id: string,
	answer: string
): Promise<Question> {
	const headers = await getHeaders();
	const url = `/adminapi/questions/${question_id}/answer`;

	const response = await fetch(url, {
		method: 'POST',
		headers,
		body: JSON.stringify({ answer })
	});

	if (!response.ok) {
		const errorBody = await response.text();
		console.error('Failed to answer question:', errorBody);
		throw new Error(`Failed to answer question: ${response.statusText}`);
	}

	return await response.json();
}

export async function addQuestion(
	question: string,
	answer?: string | null
): Promise<Question> {
	const headers = await getHeaders();
	const url = '/adminapi/questions';

	const response = await fetch(url, {
		method: 'POST',
		headers,
		body: JSON.stringify({ question, answer })
	});

	if (!response.ok) {
		const errorBody = await response.text();
		console.error('Failed to add question:', errorBody);
		throw new Error(`Failed to add question: ${response.statusText}`);
	}

	return await response.json();
}
