export interface Question {
	question_id: string;
	question: string;
	answer: string | null;
	processed: boolean;
}

export interface UpdateQuestionPayload {
	question?: string | null;
	answer?: string | null;
}

export interface IngestionJob {
	knowledgeBaseId: string;
	dataSourceId: string;
	ingestionJobId: string;
	status: string;
	createdAt: string;
	updatedAt: string;
}

export interface SyncResponse {
	status: string;
	ingestionJob: IngestionJob | null;
}
