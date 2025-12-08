/**
 * API utilities for connecting to the FastAPI backend
 */

// Add this at the top of the file
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Source {
  rank: number;
  ref: string;
  content: string;
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

export type StreamEventType = 'status' | 'sources' | 'chunk' | 'paragraph' | 'citation' | 'done' | 'error';

export interface CitationEvent {
  ref: string;
  context: string;
  hebrew: string;
  english: string;
  book?: string;
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
}

/**
 * Stream chat response from the API
 */
export async function* streamChat(
  query: string,
  useAgent: boolean = true
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_URL}/api/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      use_agent: useAgent,
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
