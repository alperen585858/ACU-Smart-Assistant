"use client";

import { useState, useRef, useEffect } from "react";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [input]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Simüle yanıt (backend bağlanınca buraya API çağrısı gelecek)
    await new Promise((r) => setTimeout(r, 800 + Math.random() * 700));

    const assistantMessage: Message = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: `Preparing a response for your question. Real responses will appear here once the backend is connected.`,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, assistantMessage]);
    setIsLoading(false);
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Top-left logo */}
      <a
        href="/"
        aria-label="Go to Home"
        className="fixed left-5 top-5 z-30 flex items-center"
      >
        <img src="/logo.svg" alt="ACU" className="h-9 w-auto" />
      </a>

      {/* Background */}
      <div
        className="pointer-events-none fixed inset-0 overflow-hidden"
        aria-hidden
      >
        <div
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: `url("/kampus.png.webp")` }}
        />
        <div className="absolute inset-0 bg-[#04032a]/65" />
      </div>

      {/* Center card */}
      <div className="relative z-10 flex min-h-screen items-center justify-center px-4 py-10">
        <div className="w-full max-w-3xl overflow-hidden rounded-3xl border border-white/30 bg-white/75 backdrop-blur-xl shadow-[0_20px_60px_rgba(0,0,0,0.35)]">
          <div className="px-6 py-4 sm:px-8">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="h-7 w-px bg-white/40" />
                <h1 className="font-semibold text-[#0b2e3b]">
                  ACU Smart Assistant
                </h1>
              </div>
              <div className="hidden text-xs font-medium text-[#0b2e3b]/60 sm:block">
                Campus knowledge chat
              </div>
            </div>
          </div>

          <div className="h-px bg-white/40" />

          <div className="flex h-[80vh] min-h-[560px] flex-col">
            {/* Messages */}
            <main className="flex-1 overflow-y-auto px-4 py-6 sm:px-8">
              {messages.length === 0 ? (
                <div className="flex h-full min-h-[52vh] flex-col items-center justify-center gap-5 text-center">
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
                      Hi, how can I help?
                    </p>
                    <p className="mt-2 max-w-sm text-sm text-[#0b2e3b]/65">
                      Type in the box below and press Enter or click Send.
                    </p>
                  </div>

                  <div className="flex flex-wrap justify-center gap-2">
                    {[
                      "Ask for information about Acibadem University",
                      "Ask about Acibadem University's departments",
                      "Get information about the Acibadem University campus",
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
                      </div>
                    </div>
                  ))}

                  {isLoading && (
                    <div className="flex justify-start animate-fade-in">
                      <div className="flex items-center gap-2 rounded-3xl bg-white/60 px-4 py-3 ring-1 ring-white/40">
                        <span className="flex h-2 w-2 animate-bounce rounded-full bg-cyan-600 [animation-delay:-0.3s]" />
                        <span className="flex h-2 w-2 animate-bounce rounded-full bg-cyan-600 [animation-delay:-0.15s]" />
                        <span className="flex h-2 w-2 animate-bounce rounded-full bg-cyan-600" />
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              )}
            </main>

            {/* Input */}
            <div className="border-t border-white/40 bg-white/55 px-4 py-4 sm:px-8">
              <form onSubmit={handleSubmit} className="flex items-end gap-3">
                <div className="relative flex-1">
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSubmit(e);
                      }
                    }}
                    placeholder="Type your message..."
                    rows={1}
                    className="w-full resize-none rounded-2xl border border-white/50 bg-white/70 px-4 py-3.5 pr-12 text-[15px] text-[#0b2e3b] placeholder:text-[#0b2e3b]/40 outline-none transition focus:border-cyan-300/60 focus:ring-2 focus:ring-cyan-300/30"
                    disabled={isLoading}
                  />

                  <button
                    type="submit"
                    disabled={!input.trim() || isLoading}
                    className="absolute bottom-2.5 right-2.5 flex h-9 w-9 items-center justify-center rounded-xl bg-cyan-600 text-white shadow-lg shadow-cyan-600/20 transition hover:bg-cyan-500 disabled:pointer-events-none disabled:opacity-40"
                    aria-label="Send"
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
  );
}
