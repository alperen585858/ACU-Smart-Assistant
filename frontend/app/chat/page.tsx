"use client";

import Image from "next/image";
import Link from "next/link";
import { useState, useRef, useEffect, useMemo, useCallback } from "react";

type RagSource = {
  url: string;
  title: string;
  cosine_distance: number;
};

type RagMeta = {
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

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  rag?: RagMeta;
  /** Assistant only: seconds from user send to reply received */
  latencySec?: number;
};

type SessionMeta = {
  id: string;
  title: string;
  updatedAt: number;
};

const CLIENT_ID_KEY = "acu-chat-client-id";
const LAST_SESSION_KEY = "acu-chat-last-session-id";

function getOrCreateClientId(): string {
  if (typeof window === "undefined") return "";
  let id = localStorage.getItem(CLIENT_ID_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(CLIENT_ID_KEY, id);
  }
  return id;
}

export default function ChatPage() {
  const [hydrated, setHydrated] = useState(false);
  const [clientId, setClientId] = useState("");
  const [sessionList, setSessionList] = useState<SessionMeta[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [typingElapsedSec, setTypingElapsedSec] = useState(0);
  const [historyOpen, setHistoryOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const requestStartRef = useRef<number | null>(null);

  const apiBase = (
    process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"
  ).replace(/\/$/, "");

  const refreshSessionList = useCallback(async () => {
    const cid = getOrCreateClientId();
    if (!cid) return;
    const r = await fetch(
      `${apiBase}/api/chat/sessions/?client_id=${encodeURIComponent(cid)}`
    );
    if (!r.ok) return;
    const data = (await r.json()) as {
      sessions: Array<{ id: string; title: string; updated_at: string }>;
    };
    setSessionList(
      data.sessions.map((s) => ({
        id: s.id,
        title: s.title,
        updatedAt: new Date(s.updated_at).getTime(),
      }))
    );
  }, [apiBase]);

  const loadMessages = useCallback(
    async (sessionId: string): Promise<boolean> => {
      const cid = getOrCreateClientId();
      const r = await fetch(
        `${apiBase}/api/chat/sessions/${sessionId}/?client_id=${encodeURIComponent(cid)}`
      );
      if (!r.ok) {
        return false;
      }
      try {
        const data = (await r.json()) as {
          messages: Array<{
            id: string;
            role: "user" | "assistant";
            content: string;
            timestamp: string;
          }>;
        };
        setMessages(
          data.messages.map((m) => ({
            id: m.id,
            role: m.role,
            content: m.content,
            timestamp: new Date(m.timestamp),
          }))
        );
        return true;
      } catch {
        return false;
      }
    },
    [apiBase]
  );

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const cid = getOrCreateClientId();
      if (cancelled) return;
      setClientId(cid);

      const r = await fetch(
        `${apiBase}/api/chat/sessions/?client_id=${encodeURIComponent(cid)}`
      );
      if (cancelled) return;
      if (r.ok) {
        const data = (await r.json()) as {
          sessions: Array<{ id: string; title: string; updated_at: string }>;
        };
        const list = data.sessions.map((s) => ({
          id: s.id,
          title: s.title,
          updatedAt: new Date(s.updated_at).getTime(),
        }));
        setSessionList(list);

        const last = localStorage.getItem(LAST_SESSION_KEY);
        if (last && list.some((s) => s.id === last)) {
          setActiveId(last);
          const okLoad = await loadMessages(last);
          if (!okLoad) {
            setMessages([]);
            setActiveId(null);
            localStorage.removeItem(LAST_SESSION_KEY);
          }
        } else {
          setActiveId(null);
          setMessages([]);
        }
      } else {
        setSessionList([]);
        setActiveId(null);
        setMessages([]);
      }
      if (!cancelled) setHydrated(true);
    })();
    return () => {
      cancelled = true;
    };
  }, [apiBase, loadMessages]);

  const activeTitle = useMemo(() => {
    if (!activeId) return "Yeni sohbet";
    return sessionList.find((s) => s.id === activeId)?.title ?? "Sohbet";
  }, [activeId, sessionList]);

  const sortedSessions = useMemo(
    () => [...sessionList].sort((a, b) => b.updatedAt - a.updatedAt),
    [sessionList]
  );

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, activeId, isLoading, typingElapsedSec]);

  useEffect(() => {
    if (!isLoading) {
      setTypingElapsedSec(0);
      return;
    }
    const start = requestStartRef.current ?? performance.now();
    requestStartRef.current = start;
    const tick = () => {
      setTypingElapsedSec((performance.now() - start) / 1000);
    };
    tick();
    const id = window.setInterval(tick, 100);
    return () => window.clearInterval(id);
  }, [isLoading]);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [input]);

  const selectSession = useCallback(
    async (id: string) => {
      setActiveId(id);
      localStorage.setItem(LAST_SESSION_KEY, id);
      const okLoad = await loadMessages(id);
      if (!okLoad) {
        setMessages([]);
      }
      if (
        typeof window !== "undefined" &&
        window.matchMedia("(max-width: 639px)").matches
      ) {
        setHistoryOpen(false);
      }
    },
    [loadMessages]
  );

  const newChat = useCallback(() => {
    setActiveId(null);
    setMessages([]);
    localStorage.removeItem(LAST_SESSION_KEY);
    if (
      typeof window !== "undefined" &&
      window.matchMedia("(max-width: 639px)").matches
    ) {
      setHistoryOpen(false);
    }
  }, []);

  const deleteSession = useCallback(
    async (id: string, e: React.MouseEvent) => {
      e.stopPropagation();
      const cid = getOrCreateClientId();
      await fetch(
        `${apiBase}/api/chat/sessions/${id}/?client_id=${encodeURIComponent(cid)}`,
        { method: "DELETE" }
      );
      await refreshSessionList();
      if (activeId === id) {
        newChat();
      }
    },
    [apiBase, activeId, newChat, refreshSessionList]
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    // State gecikmesi yüzünden gönderimin sessizce iptal olmaması için her zaman buradan al.
    const cid = getOrCreateClientId();
    if (!cid) return;
    if (!clientId) setClientId(cid);

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
      timestamp: new Date(),
    };
    setInput("");
    requestStartRef.current = performance.now();
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    const chatTimeoutMs = Number(
      process.env.NEXT_PUBLIC_CHAT_TIMEOUT_MS ?? "200000"
    );
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), chatTimeoutMs);

    const body: Record<string, string> = {
      client_id: cid,
      message: trimmed,
    };
    if (activeId) body.session_id = activeId;

    let ok = false;
    let replyText = "";
    let sidOut: string | null = null;
    let data: {
      reply?: string;
      error?: string;
      session_id?: string;
      rag?: RagMeta;
    } = {};
    try {
      const res = await fetch(`${apiBase}/api/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      try {
        data = (await res.json()) as typeof data;
      } catch {
        replyText = `Sunucu yanıtı okunamadı (HTTP ${res.status}).`;
        data = {};
      }
      if (res.ok && data.reply) {
        ok = true;
        replyText = data.reply;
        sidOut = data.session_id ?? activeId ?? null;
      } else {
        replyText =
          data.error ||
          replyText ||
          `Sunucu hatası (${res.status}). Ollama ve veritabanı çalışıyor mu kontrol edin.`;
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        replyText = `Yanıt ${Math.round(chatTimeoutMs / 1000)} sn içinde gelmedi.`;
      } else {
        replyText =
          "Bağlantı kurulamadı. Backend adresini (NEXT_PUBLIC_API_URL) kontrol edin.";
      }
    } finally {
      window.clearTimeout(timeoutId);
    }

    const latencySec =
      requestStartRef.current != null
        ? Math.round(((performance.now() - requestStartRef.current) / 1000) * 10) / 10
        : undefined;
    requestStartRef.current = null;

    if (ok && sidOut) {
      setActiveId(sidOut);
      localStorage.setItem(LAST_SESSION_KEY, sidOut);
      const loaded = await loadMessages(sidOut);
      if (!loaded) {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: replyText,
            timestamp: new Date(),
            rag: data.rag,
            latencySec,
          },
        ]);
      } else if (data.rag) {
        setMessages((prev) => {
          if (prev.length === 0) return prev;
          const last = prev[prev.length - 1];
          if (last.role !== "assistant") return prev;
          return [
            ...prev.slice(0, -1),
            { ...last, rag: data.rag, latencySec: latencySec ?? last.latencySec },
          ];
        });
      } else if (latencySec != null) {
        setMessages((prev) => {
          if (prev.length === 0) return prev;
          const last = prev[prev.length - 1];
          if (last.role !== "assistant") return prev;
          return [...prev.slice(0, -1), { ...last, latencySec }];
        });
      }
    } else {
      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: replyText,
        timestamp: new Date(),
        rag: data.rag,
        latencySec,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    }

    await refreshSessionList();
    setIsLoading(false);
  };

  if (!hydrated) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#0b2e3b]/10 text-[#0b2e3b]">
        Yükleniyor…
      </div>
    );
  }

  return (
    <div className="relative min-h-screen overflow-hidden">
      <Link
        href="/"
        aria-label="Ana sayfa"
        className="fixed left-5 top-5 z-30 flex items-center"
      >
        <Image
          src="/logo.svg"
          alt="ACU"
          width={120}
          height={36}
          className="h-9 w-auto"
          unoptimized
        />
      </Link>

      <div
        className="pointer-events-none fixed inset-0 overflow-hidden"
        aria-hidden
      >
        <div className="absolute inset-0 bg-gradient-to-br from-[#0b2e3b] via-[#0a1628] to-[#162a3e]" />
        <div className="absolute -top-32 -right-32 h-96 w-96 rounded-full bg-cyan-500/10 blur-3xl animate-float" />
        <div className="absolute -bottom-32 -left-32 h-96 w-96 rounded-full bg-blue-500/10 blur-3xl animate-float [animation-delay:3s]" />
      </div>

      <div className="relative z-10 flex min-h-screen items-center justify-center px-4 py-10 pt-20 sm:pt-10">
        <div className="flex h-[calc(100dvh-5rem)] w-full max-w-4xl flex-col overflow-hidden rounded-3xl border border-white/30 bg-white/75 shadow-[0_20px_60px_rgba(0,0,0,0.35)] backdrop-blur-xl sm:h-[85vh] sm:min-h-[560px]">
          <div className="flex shrink-0 items-center gap-2 border-b border-white/40 px-3 py-3 sm:gap-3 sm:px-6">
            <button
              type="button"
              onClick={() => setHistoryOpen((v) => !v)}
              aria-expanded={historyOpen}
              aria-controls="chat-history-panel"
              title={historyOpen ? "Sohbet listesini gizle" : "Sohbet geçmişi"}
              className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border transition ${
                historyOpen
                  ? "border-cyan-500/50 bg-cyan-500/15 text-cyan-800"
                  : "border-white/50 bg-white/50 text-[#0b2e3b] hover:bg-white/70"
              }`}
            >
              <svg
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.75}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z"
                />
              </svg>
            </button>

            <div className="h-6 w-px shrink-0 bg-[#0b2e3b]/20" />

            <div className="min-w-0 flex-1">
              <h1 className="truncate text-sm font-semibold text-[#0b2e3b] sm:text-base">
                ACU Smart Assistant
              </h1>
              <p className="truncate text-xs text-[#0b2e3b]/55 sm:hidden">
                {activeTitle}
              </p>
            </div>

            <span className="hidden shrink-0 text-xs font-medium text-[#0b2e3b]/55 sm:block">
              Veritabanında saklanır
            </span>
          </div>

          <div className="flex min-h-0 flex-1 overflow-hidden">
            <div
              id="chat-history-panel"
              className={`flex shrink-0 flex-col overflow-hidden border-[#0b2e3b]/10 bg-white/45 transition-[width,opacity] duration-200 ease-out ${
                historyOpen
                  ? "w-[min(13.5rem,42vw)] border-r opacity-100"
                  : "w-0 border-0 opacity-0"
              }`}
            >
              <div className="flex h-full w-[min(13.5rem,42vw)] flex-col">
                <button
                  type="button"
                  onClick={newChat}
                  className="mx-2 mt-2 flex items-center justify-center gap-1.5 rounded-xl border border-white/60 bg-white/60 py-2 text-xs font-medium text-[#0b2e3b] shadow-sm transition hover:bg-white/90 sm:text-sm"
                >
                  <svg
                    className="h-4 w-4 shrink-0"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M12 4v16m8-8H4"
                    />
                  </svg>
                  Yeni sohbet
                </button>

                <nav className="mt-2 flex-1 overflow-y-auto px-2 pb-3">
                  <p className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-wide text-[#0b2e3b]/45">
                    Geçmiş (sunucu)
                  </p>
                  <ul className="space-y-0.5">
                    {sortedSessions.map((s) => (
                      <li key={s.id}>
                        <div
                          role="button"
                          tabIndex={0}
                          onClick={() => selectSession(s.id)}
                          onKeyDown={(ev) => {
                            if (ev.key === "Enter" || ev.key === " ") {
                              ev.preventDefault();
                              void selectSession(s.id);
                            }
                          }}
                          className={`group flex w-full items-center gap-0.5 rounded-lg px-2 py-1.5 text-left text-xs transition sm:text-[13px] ${
                            s.id === activeId
                              ? "bg-cyan-600/15 font-medium text-[#0b2e3b] ring-1 ring-cyan-600/25"
                              : "text-[#0b2e3b]/80 hover:bg-white/50"
                          }`}
                        >
                          <span className="min-w-0 flex-1 truncate" title={s.title}>
                            {s.title}
                          </span>
                          <button
                            type="button"
                            onClick={(ev) => void deleteSession(s.id, ev)}
                            className="shrink-0 rounded-md p-0.5 text-[#0b2e3b]/35 opacity-0 transition hover:bg-red-500/15 hover:text-red-700 group-hover:opacity-100"
                            aria-label="Sohbeti sil"
                          >
                            <svg
                              className="h-3.5 w-3.5 sm:h-4 sm:w-4"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                              strokeWidth={1.5}
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"
                              />
                            </svg>
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                </nav>
              </div>
            </div>

            <div className="flex min-h-0 min-w-0 flex-1 flex-col">
              <main className="min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-8">
                {messages.length === 0 ? (
                  <div className="flex h-full min-h-[42vh] flex-col items-center justify-center gap-5 text-center">
                    <div className="flex h-20 w-20 items-center justify-center rounded-3xl bg-gradient-to-br from-cyan-500/20 to-blue-500/15 ring-1 ring-white/40">
                      <svg
                        className="h-10 w-10 text-cyan-600"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={1.5}
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z"
                        />
                      </svg>
                    </div>

                    <div>
                      <p className="text-xl font-semibold text-[#0b2e3b]">
                        Merhaba, nasıl yardımcı olabilirim?
                      </p>
                      <p className="mt-2 max-w-sm text-sm text-[#0b2e3b]/65">
                        Acıbadem Üniversitesi hakkında sorularınızı yanıtlıyorum.
                      </p>
                    </div>

                    <div className="flex flex-wrap justify-center gap-2">
                      {[
                        "Acıbadem Üniversitesi hakkında bilgi ver",
                        "Bölümler nelerdir?",
                        "Kampüs hakkında kısa özet",
                      ].map((suggestion) => (
                        <button
                          key={suggestion}
                          type="button"
                          onClick={() => setInput(suggestion)}
                          className="rounded-full border border-white/40 bg-white/40 px-4 py-2 text-sm text-[#0b2e3b]/80 transition-colors hover:bg-white/65"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {messages.map((msg) => (
                      <div
                        key={msg.id}
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
                          {msg.role === "assistant" &&
                            typeof msg.latencySec === "number" && (
                              <p className="mt-2 flex items-center gap-1.5 text-[11px] font-medium tabular-nums tracking-tight text-[#0b2e3b]/45">
                                <span
                                  className="inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500/80"
                                  aria-hidden
                                />
                                Yanıt süresi{" "}
                                <span className="rounded-md bg-[#0b2e3b]/[0.06] px-1.5 py-0.5 font-mono text-[#0b2e3b]/70">
                                  {msg.latencySec.toFixed(1)} sn
                                </span>
                              </p>
                            )}
                          {msg.role === "assistant" && msg.rag && (
                            <div className="mt-2 border-t border-[#0b2e3b]/12 pt-2 text-left text-[11px] leading-snug text-[#0b2e3b]/60">
                              {typeof msg.rag.indexed_chunks_in_db === "number" ? (
                                <p className="mb-2 rounded-md bg-[#0b2e3b]/5 px-2 py-1.5 font-mono text-[10px] text-[#0b2e3b]/75">
                                  <span className="font-semibold text-[#0b2e3b]/55">
                                    Veri / model görünürlüğü:{" "}
                                  </span>
                                  DB&apos;de{" "}
                                  <strong>{msg.rag.indexed_chunks_in_db}</strong> chunk · Bu
                                  istekte LLM&apos;e giden bağlam{" "}
                                  <strong>{msg.rag.context_chars_sent ?? 0}</strong> karakter ·
                                  Son kullanıcı mesajı (LLM) toplam{" "}
                                  <strong>{msg.rag.llm_user_turn_chars ?? 0}</strong> karakter ·
                                  Bağlam LLM mesajında:{" "}
                                  <strong>
                                    {msg.rag.context_block_in_llm ? "var" : "yok"}
                                  </strong>
                                </p>
                              ) : null}
                              {msg.rag.chunks_used === 0 ? (
                                <p className="text-amber-800/90">
                                  Taranan siteden bu soruya yakın metin gelmedi (
                                  <code className="rounded bg-[#0b2e3b]/5 px-1">
                                    chunks_used=0
                                  </code>
                                  ). Yanıt genel bilgiye dayanıyor olabilir;{" "}
                                  <code className="rounded bg-[#0b2e3b]/5 px-1">
                                    refresh_rag
                                  </code>{" "}
                                  çalıştırın veya soruyu İngilizce/üniversite adıyla tekrar deneyin.
                                </p>
                              ) : (
                                <>
                                  <p className="mb-1 font-semibold uppercase tracking-wide text-[#0b2e3b]/50">
                                    Modele gönderilen kaynaklar
                                  </p>
                                  <ul className="space-y-1">
                                    {msg.rag.sources.map((s, i) => (
                                      <li key={`${s.url}-${i}`}>
                                        <a
                                          href={s.url}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="font-medium text-cyan-700 underline decoration-cyan-700/30 underline-offset-2 hover:text-cyan-600"
                                        >
                                          {s.title?.trim() || s.url}
                                        </a>
                                        <span className="ml-1 opacity-75">
                                          (kosinüs mesafesi {s.cosine_distance})
                                        </span>
                                      </li>
                                    ))}
                                  </ul>
                                  {msg.rag.relaxed_retrieval ? (
                                    <p className="mt-1.5 italic text-[#0b2e3b]/55">
                                      Sıkı eşik altında eşleşme yoktu; en yakın parçalar
                                      kullanıldı.
                                    </p>
                                  ) : null}
                                  {!msg.rag.embedding_ok ? (
                                    <p className="mt-1 text-amber-800/90">
                                      Embedding üretilemedi; RAG devre dışı kaldı.
                                    </p>
                                  ) : null}
                                </>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}

                    {isLoading && (
                      <div className="flex justify-start animate-fade-in">
                        <div className="flex items-center gap-4 rounded-3xl border border-white/35 bg-gradient-to-r from-white/70 to-white/55 px-5 py-3.5 shadow-sm ring-1 ring-cyan-500/10 backdrop-blur-sm">
                          <div className="flex items-center gap-1.5" aria-hidden>
                            <span className="h-2 w-2 animate-bounce rounded-full bg-gradient-to-br from-cyan-500 to-cyan-600 [animation-delay:-0.32s]" />
                            <span className="h-2 w-2 animate-bounce rounded-full bg-gradient-to-br from-cyan-500 to-cyan-600 [animation-delay:-0.16s]" />
                            <span className="h-2 w-2 animate-bounce rounded-full bg-gradient-to-br from-cyan-500 to-cyan-600" />
                          </div>
                          <div className="flex min-w-[5.5rem] flex-col leading-none">
                            <span className="text-[10px] font-semibold uppercase tracking-wider text-[#0b2e3b]/40">
                              Yanıt bekleniyor
                            </span>
                            <span className="mt-1 font-mono text-lg font-semibold tabular-nums tracking-tight text-cyan-800">
                              {typingElapsedSec.toFixed(1)}
                              <span className="ml-0.5 text-sm font-medium text-[#0b2e3b]/45">
                                sn
                              </span>
                            </span>
                          </div>
                        </div>
                      </div>
                    )}

                    <div ref={messagesEndRef} />
                  </div>
                )}
              </main>

              <div className="shrink-0 border-t border-white/40 bg-white/55 px-4 py-4 sm:px-8">
                <form onSubmit={handleSubmit} className="flex items-end gap-3">
                  <div className="relative flex-1">
                    <textarea
                      ref={textareaRef}
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(ev) => {
                        if (ev.key === "Enter" && !ev.shiftKey) {
                          ev.preventDefault();
                          handleSubmit(ev);
                        }
                      }}
                      placeholder="Mesajınızı yazın..."
                      rows={1}
                      className="w-full resize-none rounded-2xl border border-white/50 bg-white/70 px-4 py-3.5 pr-12 text-[15px] text-[#0b2e3b] placeholder:text-[#0b2e3b]/40 outline-none transition focus:border-cyan-300/60 focus:ring-2 focus:ring-cyan-300/30"
                      disabled={isLoading}
                    />

                    <button
                      type="submit"
                      disabled={!input.trim() || isLoading}
                      className="absolute bottom-2.5 right-2.5 flex h-9 w-9 items-center justify-center rounded-xl bg-cyan-600 text-white shadow-lg shadow-cyan-600/20 transition hover:bg-cyan-500 disabled:pointer-events-none disabled:opacity-40"
                      aria-label="Gönder"
                    >
                      <svg
                        className="h-4 w-4"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={2}
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                        />
                      </svg>
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
