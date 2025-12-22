/**
 * API utilities for connecting to the FastAPI backend
 */

// API URL - defaults to localhost:8001 for local development
// Set NEXT_PUBLIC_API_URL in .env.local to override (e.g., for deployed backend)
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export interface Source {
  rank: number;
  ref: string;
  content: string;
  context_text?: string;  // Contextual notes from embedding
  similarity: number;
  rerank_score?: number;
  book: string;
  siman?: number;
  seif?: number;
  commentaries: Array<{
    commentator: string;
    text: string;
  }>;
  hydrated: boolean;
}

export interface SefariaVerse {
  number: number;
  he: string;
  en: string;
  isTarget: boolean;
}

export interface SefariaText {
  ref: string;
  title: string;
  heTitle: string;
  verses: SefariaVerse[];
  targetVerse?: number;
}

export type StreamEventType = 'status' | 'sources' | 'sources_only' | 'chunk' | 'paragraph' | 'citation' | 'done' | 'error' | 'metrics' | 'metrics_summary';

export interface CitationEvent {
  ref: string;
  context: string;       // Now: Mistral's translation
  hebrew: string;
  english: string;       // Sefaria's translation (reference)
  book?: string;
  hebrew_highlight?: string;  // The specific Hebrew portion being translated
  translation_success?: boolean;  // Whether the dedicated translation succeeded
}

// Performance metrics types
export interface MetricsDetails {
  dimensions?: number;
  chunks_found?: number;
  similarity_max?: number;
  similarity_avg?: number;
  was_reranked?: boolean;
  chunks_after?: number;
  top_rerank_score?: number;
  chunks_hydrated?: number;
  commentaries_fetched?: number;
  paragraphs?: number;
  citations?: number;
  approx_words?: number;
  words_per_second?: number;
}

export interface MetricsBreakdown {
  embedding?: number;
  vector_search?: number;
  reranking?: number;
  hydration?: number;
  source_cache?: number;  // Time to fetch translations from Sefaria
  time_to_first_token?: number;
  llm_generation?: number;
  total_pipeline?: number;
}

export interface MetricsCounts {
  embedding_dimensions?: number;
  chunks_retrieved?: number;
  chunks_after_rerank?: number;
  chunks_hydrated?: number;
  commentaries_fetched?: number;
  context_length?: number;
  paragraphs?: number;
  citations?: number;
  approx_words?: number;
}

export interface MetricsMetadata {
  similarity_max?: number;
  similarity_min?: number;
  similarity_avg?: number;
  reranking_applied?: boolean;
  rerank_score_max?: number;
  rerank_score_avg?: number;
  words_per_second?: number;
}

export interface MetricsSummary {
  total_duration_ms: number;
  breakdown: MetricsBreakdown;
  counts: MetricsCounts;
  metadata: MetricsMetadata;
}

export interface StreamEvent {
  type: StreamEventType;
  message?: string;
  content?: string;
  data?: Source[];
  // Citation-specific fields
  ref?: string;
  context?: string;
  hebrew?: string;
  english?: string;
  book?: string;
  hebrew_highlight?: string;  // The specific Hebrew portion being translated
  translation_success?: boolean;  // Whether the dedicated translation succeeded
  // Metrics-specific fields
  stage?: string;
  duration_ms?: number;
  details?: MetricsDetails;
  // Metrics summary fields
  total_duration_ms?: number;
  breakdown?: MetricsBreakdown;
  counts?: MetricsCounts;
  metadata?: MetricsMetadata;
}

// Frontend performance tracking
export interface FrontendMetrics {
  queryStartTime: number;
  timeToFirstEvent?: number;
  timeToFirstContent?: number;
  timeToSources?: number;
  timeToComplete?: number;
  eventCount: number;
  networkLatencyEstimate?: number;
}

/**
 * Stream chat response from the API
 */
export async function* streamChat(
  query: string,
  useAgent: boolean = true,
  sourcesOnly: boolean = false
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_URL}/api/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      use_agent: useAgent,
      sources_only: sourcesOnly,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          yield data as StreamEvent;
        } catch {
          // Ignore parse errors
        }
      }
    }
  }
}

/**
 * Fetch Sefaria text for sidebar
 */
export async function fetchSefariaText(ref: string): Promise<SefariaText> {
  const encodedRef = encodeURIComponent(ref);
  const response = await fetch(`${API_URL}/api/sefaria/text/${encodedRef}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch text: ${response.status}`);
  }

  return response.json();
}
