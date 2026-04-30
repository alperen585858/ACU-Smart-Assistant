"use client";

import Image from "next/image";
import Link from "next/link";
import { useState, useRef, useEffect, useMemo, useCallback } from "react";

import type { Message, SessionMeta, RagMeta } from "./types";
import ChatMessage from "./_components/ChatMessage";
import SessionSidebar from "./_components/SessionSidebar";
import TypingIndicator from "./_components/TypingIndicator";
import EmptyState from "./_components/EmptyState";
import ChatInput from "./_components/ChatInput";

const CLIENT_ID_KEY = "acu-chat-client-id";
const LAST_SESSION_KEY = "acu-chat-last-session-id";
const DRAFT_SESSION_KEY = "__draft__";

/** RFC4122 v4; works on http://IP where `crypto.randomUUID` is missing. */
function newUuid(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  const bytes = new Uint8Array(16);
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256);
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const h = [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

function getOrCreateClientId(): string {
  if (typeof window === "undefined") return "";
  let id = localStorage.getItem(CLIENT_ID_KEY);
  if (!id) {
    id = newUuid();
    localStorage.setItem(CLIENT_ID_KEY, id);
  }
  return id;
}

export default function ChatPage() {
  /* ── state ── */
  const [hydrated, setHydrated] = useState(false);
  const [clientId, setClientId] = useState("");
  const [sessionList, setSessionList] = useState<SessionMeta[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingSessionKey, setLoadingSessionKey] = useState<string | null>(null);
  const [typingElapsedSec, setTypingElapsedSec] = useState(0);
  const [historyOpen, setHistoryOpen] = useState(false);

  /* ── refs ── */
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesScrollRef = useRef<HTMLElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const requestStartRef = useRef<number | null>(null);
  const activeIdRef = useRef<string | null>(null);
  const activeChatAbortRef = useRef<AbortController | null>(null);
  const abortReasonRef = useRef<"timeout" | "user" | null>(null);

  const apiBase = (
    process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"
  ).replace(/\/$/, "");

  /* ── derived ── */
  const viewSessionKey = activeId ?? DRAFT_SESSION_KEY;
  const typingForThisView = Boolean(
    isLoading && loadingSessionKey === viewSessionKey
  );
  const activeTitle = useMemo(() => {
    if (!activeId) return "New chat";
    return sessionList.find((s) => s.id === activeId)?.title ?? "Chat";
  }, [activeId, sessionList]);
  const sortedSessions = useMemo(
    () => [...sessionList].sort((a, b) => b.updatedAt - a.updatedAt),
    [sessionList]
  );

  /* ── API helpers ── */
  const refreshSessionList = useCallback(async () => {
    const cid = getOrCreateClientId();
    if (!cid) return;
    try {
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
    } catch {
      // Backend unreachable
    }
  }, [apiBase]);

  const loadMessages = useCallback(
    async (sessionId: string): Promise<boolean> => {
      const cid = getOrCreateClientId();
      const r = await fetch(
        `${apiBase}/api/chat/sessions/${sessionId}/?client_id=${encodeURIComponent(cid)}`
      );
      if (!r.ok) return false;
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

  /* ── hydration + initial load ── */
  useEffect(() => {
    let cancelled = false;
    let backupTimer: number | null = window.setTimeout(() => {
      if (!cancelled) setHydrated(true);
    }, 20_000);
    const clearBackup = () => {
      if (backupTimer != null) {
        clearTimeout(backupTimer);
        backupTimer = null;
      }
    };
    (async () => {
      const cid = getOrCreateClientId();
      if (cancelled) return;
      setClientId(cid);
      try {
        const r = await fetch(
          `${apiBase}/api/chat/sessions/?client_id=${encodeURIComponent(cid)}`,
          { signal: AbortSignal.timeout(18_000) }
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
      } catch {
        if (!cancelled) {
          setSessionList([]);
          setActiveId(null);
          setMessages([]);
        }
      }
      if (!cancelled) {
        clearBackup();
        setHydrated(true);
      }
    })();
    return () => {
      cancelled = true;
      clearBackup();
    };
  }, [apiBase, loadMessages]);

  /* ── sync activeId ref ── */
  useEffect(() => {
    activeIdRef.current = activeId;
  }, [activeId]);

  /* ── scrolling ── */
  const updateAutoScrollState = useCallback(() => {
    const scroller = messagesScrollRef.current;
    if (!scroller) return;
    const dist = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
    shouldAutoScrollRef.current = dist < 64;
  }, []);

  const scrollToBottom = (behavior: ScrollBehavior = "smooth") => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  };

  useEffect(() => {
    if (!shouldAutoScrollRef.current) return;
    scrollToBottom(typingForThisView ? "auto" : "smooth");
  }, [messages, activeId, typingForThisView, typingElapsedSec]);

  useEffect(() => {
    shouldAutoScrollRef.current = true;
    requestAnimationFrame(() => {
      scrollToBottom("auto");
      updateAutoScrollState();
    });
  }, [activeId, updateAutoScrollState]);

  /* ── typing timer ── */
  useEffect(() => {
    if (!typingForThisView) {
      setTypingElapsedSec(0);
      return;
    }
    const start = requestStartRef.current ?? performance.now();
    const tick = () => setTypingElapsedSec((performance.now() - start) / 1000);
    tick();
    const id = window.setInterval(tick, 100);
    return () => window.clearInterval(id);
  }, [typingForThisView]);

  /* ── session actions ── */
  const selectSession = useCallback(
    async (id: string) => {
      setActiveId(id);
      localStorage.setItem(LAST_SESSION_KEY, id);
      const okLoad = await loadMessages(id);
      if (!okLoad) setMessages([]);
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
      if (activeId === id) newChat();
    },
    [apiBase, activeId, newChat, refreshSessionList]
  );

  const stopGeneration = useCallback(() => {
    if (!isLoading || !activeChatAbortRef.current) return;
    abortReasonRef.current = "user";
    activeChatAbortRef.current.abort();
  }, [isLoading]);

  /* ── send message ── */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const activeIdWhenSent = activeId;
    const requestOwnerKey = activeId ?? DRAFT_SESSION_KEY;
    const stillOnSentView = () => activeIdRef.current === activeIdWhenSent;

    const cid = getOrCreateClientId();
    if (!cid) return;
    if (!clientId) setClientId(cid);

    const userMessage: Message = {
      id: newUuid(),
      role: "user",
      content: trimmed,
      timestamp: new Date(),
    };
    setInput("");
    requestStartRef.current = performance.now();
    if (stillOnSentView()) setMessages((prev) => [...prev, userMessage]);
    setLoadingSessionKey(requestOwnerKey);
    setIsLoading(true);

    const controller = new AbortController();
    activeChatAbortRef.current = controller;
    abortReasonRef.current = null;

    const body: Record<string, string> = { client_id: cid, message: trimmed };
    if (activeId) body.session_id = activeId;

    let ok = false;
    let replyText = "";
    let sidOut: string | null = null;
    let userCancelled = false;
    let data: { reply?: string; error?: string; session_id?: string; rag?: RagMeta } = {};

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
        replyText = `Could not read server response (HTTP ${res.status}).`;
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
          `Server error (${res.status}). Check that Ollama and the database are running.`;
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        if (abortReasonRef.current === "user") {
          userCancelled = true;
          replyText = "Response generation was stopped.";
        } else {
          replyText = "Request was aborted before completion.";
        }
      } else {
        replyText =
          `Could not reach the API at ${apiBase} (network error). ` +
          "Is Django running? If you use Nginx on :8080, set NEXT_PUBLIC_API_URL to that " +
          "base URL in Project/.env and in frontend/.env.local, then restart the Next dev server. " +
          "If you open the app from another device, use the computer's LAN IP instead of localhost.";
      }
    } finally {
      activeChatAbortRef.current = null;
      abortReasonRef.current = null;
    }

    const latencySec =
      requestStartRef.current != null
        ? Math.round(((performance.now() - requestStartRef.current) / 1000) * 10) / 10
        : undefined;
    requestStartRef.current = null;

    if (userCancelled) {
      if (stillOnSentView()) {
        setMessages((prev) => [
          ...prev,
          { id: newUuid(), role: "assistant", content: replyText, timestamp: new Date() },
        ]);
      }
      await refreshSessionList();
      setLoadingSessionKey(null);
      setIsLoading(false);
      return;
    }

    if (ok && sidOut) {
      if (stillOnSentView()) {
        setActiveId(sidOut);
        localStorage.setItem(LAST_SESSION_KEY, sidOut);
        const loaded = await loadMessages(sidOut);
        if (!loaded) {
          setMessages((prev) => [
            ...prev,
            { id: newUuid(), role: "assistant", content: replyText, timestamp: new Date(), rag: data.rag, latencySec },
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
      }
    } else if (stillOnSentView()) {
      setMessages((prev) => [
        ...prev,
        { id: newUuid(), role: "assistant", content: replyText, timestamp: new Date(), rag: data.rag, latencySec },
      ]);
    }

    await refreshSessionList();
    setLoadingSessionKey(null);
    setIsLoading(false);
  };

  /* ── loading screen ── */
  if (!hydrated) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#0b2e3b]/10 text-[#0b2e3b]">
        Loading…
      </div>
    );
  }

  /* ── main render ── */
  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden" aria-hidden>
        <div
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: "url('/kampus.png.webp')" }}
        />
        <div className="absolute inset-0 bg-[#081626]/45" />
        <div className="absolute -top-32 -right-32 h-96 w-96 rounded-full bg-cyan-500/10 blur-3xl animate-float" />
        <div className="absolute -bottom-32 -left-32 h-96 w-96 rounded-full bg-blue-500/10 blur-3xl animate-float [animation-delay:3s]" />
      </div>

      <div className="relative z-10 flex min-h-screen items-center justify-center px-4 py-10 pt-20 sm:pt-10">
        <div className="flex h-[calc(100dvh-5rem)] w-full max-w-4xl flex-col overflow-hidden rounded-3xl border border-white/30 bg-white/75 shadow-[0_20px_60px_rgba(0,0,0,0.35)] backdrop-blur-xl sm:h-[85vh] sm:min-h-[560px]">

          {/* ── Header ── */}
          <div className="flex shrink-0 items-center gap-2 border-b border-white/40 px-3 py-3 sm:gap-3 sm:px-6">
            <Link href="/" aria-label="Home" className="flex items-center pr-1">
              <Image src="/logo.svg" alt="ACU" width={110} height={34} className="h-8 w-auto" unoptimized />
            </Link>
            <div className="h-6 w-px shrink-0 bg-[#0b2e3b]/20" />
            <button
              type="button"
              onClick={() => setHistoryOpen((v) => !v)}
              aria-expanded={historyOpen}
              aria-controls="chat-history-panel"
              title={historyOpen ? "Hide conversation list" : "Conversation history"}
              className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border transition ${
                historyOpen
                  ? "border-cyan-500/50 bg-cyan-500/15 text-cyan-800"
                  : "border-white/50 bg-white/50 text-[#0b2e3b] hover:bg-white/70"
              }`}
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
              </svg>
            </button>
            <div className="h-6 w-px shrink-0 bg-[#0b2e3b]/20" />
            <div className="min-w-0 flex-1">
              <h1 className="truncate text-sm font-semibold text-[#0b2e3b] sm:text-base">
                ACU Smart Assistant
              </h1>
              <p className="truncate text-xs text-[#0b2e3b]/55 sm:hidden">{activeTitle}</p>
            </div>
          </div>

          {/* ── Body ── */}
          <div className="flex min-h-0 flex-1 overflow-hidden">
            <SessionSidebar
              sessions={sortedSessions}
              activeId={activeId}
              isOpen={historyOpen}
              onSelect={selectSession}
              onNewChat={newChat}
              onDelete={deleteSession}
            />

            <div className="flex min-h-0 min-w-0 flex-1 flex-col">
              <main
                ref={messagesScrollRef}
                onScroll={updateAutoScrollState}
                className="min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-8"
              >
                {messages.length === 0 ? (
                  <EmptyState onSuggestionClick={setInput} />
                ) : (
                  <div className="space-y-4">
                    {messages.map((msg) => (
                      <ChatMessage key={msg.id} message={msg} />
                    ))}
                    {typingForThisView && (
                      <TypingIndicator
                        elapsedSec={typingElapsedSec}
                        onStop={stopGeneration}
                      />
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                )}
              </main>

              <ChatInput
                value={input}
                onChange={setInput}
                onSubmit={handleSubmit}
                onStop={stopGeneration}
                isLoading={typingForThisView}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
