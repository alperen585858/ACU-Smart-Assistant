export type RagSource = {
  url: string;
  title: string;
  cosine_distance: number;
};

export type RagMeta = {
  embedding_ok: boolean;
  chunks_used: number;
  relaxed_retrieval: boolean;
  sources: RagSource[];
  rag_query_preview: string;
  reason?: string;
  /** Backend: DocumentChunk rows in DB */
  indexed_chunks_in_db?: number;
  /** Characters of crawled excerpts injected into the last user turn */
  context_chars_sent?: number;
  /** Full last user message length sent to the LLM (includes ===CONTEXT=== wrapper) */
  llm_user_turn_chars?: number;
  /** True if crawled text was embedded in the prompt */
  context_block_in_llm?: boolean;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  rag?: RagMeta;
  /** Assistant only: seconds from user send to reply received */
  latencySec?: number;
};

export type SessionMeta = {
  id: string;
  title: string;
  updatedAt: number;
};
