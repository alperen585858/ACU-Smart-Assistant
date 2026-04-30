import type { Message } from "../types";

export default function ChatMessage({ message: msg }: { message: Message }) {
  return (
    <div
      className={`flex animate-fade-in ${
        msg.role === "user" ? "justify-end" : "justify-start"
      }`}
    >
      <div
        className={`max-w-[85%] rounded-3xl px-4 py-3 sm:max-w-[75%] ${
          msg.role === "user"
            ? "bg-cyan-600 text-white shadow-lg shadow-cyan-600/20"
            : "bg-white/65 text-[#0b2e3b] ring-1 ring-white/45"
        }`}
      >
        <p className="whitespace-pre-wrap text-[15px] leading-relaxed">
          {msg.content}
        </p>

        {msg.role === "assistant" && typeof msg.latencySec === "number" && (
          <p className="mt-2 flex items-center gap-1.5 text-[11px] font-medium tabular-nums tracking-tight text-[#0b2e3b]/45">
            <span
              className="inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500/80"
              aria-hidden
            />
            Response time{" "}
            <span className="rounded-md bg-[#0b2e3b]/[0.06] px-1.5 py-0.5 font-mono text-[#0b2e3b]/70">
              {msg.latencySec.toFixed(1)}s
            </span>
          </p>
        )}

        {msg.role === "assistant" && msg.rag && (
          <div className="mt-2 border-t border-[#0b2e3b]/12 pt-2 text-left text-[11px] leading-snug text-[#0b2e3b]/60">
            {typeof msg.rag.indexed_chunks_in_db === "number" ? (
              <p className="mb-2 rounded-md bg-[#0b2e3b]/5 px-2 py-1.5 font-mono text-[10px] text-[#0b2e3b]/75">
                <span className="font-semibold text-[#0b2e3b]/55">
                  Data / model visibility:{" "}
                </span>
                <strong>{msg.rag.indexed_chunks_in_db}</strong> chunks in DB
                · Context sent to LLM this request:{" "}
                <strong>{msg.rag.context_chars_sent ?? 0}</strong> chars
                · Last user turn (LLM) total:{" "}
                <strong>{msg.rag.llm_user_turn_chars ?? 0}</strong> chars
                · Context in LLM message:{" "}
                <strong>
                  {msg.rag.context_block_in_llm ? "yes" : "no"}
                </strong>
              </p>
            ) : null}

            {msg.rag.chunks_used === 0 &&
            msg.rag.reason?.startsWith("skipped_") ? (
              <p className="text-[#0b2e3b]/50">
                Intent:{" "}
                {msg.rag.reason === "skipped_smalltalk_no_rag"
                  ? "smalltalk"
                  : "off-topic"}{" "}
                — RAG skipped.
              </p>
            ) : msg.rag.chunks_used === 0 ? (
              <p className="text-amber-800/90">
                No close match from the crawled site for this question (
                <code className="rounded bg-[#0b2e3b]/5 px-1">
                  chunks_used=0
                </code>
                ). The answer may rely on general knowledge; run{" "}
                <code className="rounded bg-[#0b2e3b]/5 px-1">
                  refresh_rag
                </code>{" "}
                or try rephrasing with the university name in English.
              </p>
            ) : (
              <>
                <p className="mb-1 font-semibold uppercase tracking-wide text-[#0b2e3b]/50">
                  Sources sent to the model
                </p>
                <ul className="space-y-1">
                  {msg.rag.sources.map((s, i) => (
                    <li key={`${s.url}-${i}`}>
                      <a
                        href={
                          s.url && /^https?:\/\//i.test(s.url) ? s.url : "#"
                        }
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-medium text-cyan-700 underline decoration-cyan-700/30 underline-offset-2 hover:text-cyan-600"
                      >
                        {s.title?.trim() || s.url}
                      </a>
                      <span className="ml-1 opacity-75">
                        (cosine distance {s.cosine_distance})
                      </span>
                    </li>
                  ))}
                </ul>
                {msg.rag.relaxed_retrieval ? (
                  <p className="mt-1.5 italic text-[#0b2e3b]/55">
                    No chunk passed the strict distance threshold; closest
                    excerpts were used instead.
                  </p>
                ) : null}
                {!msg.rag.embedding_ok ? (
                  <p className="mt-1 text-amber-800/90">
                    Embedding failed; RAG was skipped.
                  </p>
                ) : null}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
