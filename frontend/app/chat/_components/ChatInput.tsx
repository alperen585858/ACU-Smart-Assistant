"use client";

import { useEffect, useRef } from "react";

type Props = {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  onStop: () => void;
  isLoading: boolean;
};

export default function ChatInput({
  value,
  onChange,
  onSubmit,
  onStop,
  isLoading,
}: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [value]);

  return (
    <div className="shrink-0 border-t border-white/40 bg-white/55 px-4 py-4 sm:px-8">
      <form onSubmit={onSubmit} className="flex items-end gap-3">
        <div className="relative flex-1">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(ev) => {
              if (ev.key === "Enter" && !ev.shiftKey) {
                ev.preventDefault();
                onSubmit(ev);
              }
            }}
            placeholder="Type your message…"
            rows={1}
            className="chat-input w-full resize-none rounded-2xl border border-white/50 bg-white/70 px-4 py-3.5 pr-12 text-[15px] text-[#0b2e3b] placeholder:text-[#0b2e3b]/40 outline-none transition focus:border-cyan-300/60 focus:ring-2 focus:ring-cyan-300/30"
            disabled={isLoading}
          />

          {isLoading ? (
            <button
              type="button"
              onClick={onStop}
              title="Stop"
              aria-label="Stop generating response"
              className="absolute right-2.5 top-1/2 -translate-y-1/2 flex h-9 w-9 items-center justify-center rounded-lg border border-[#0b2e3b]/20 bg-[#0b2e3b] text-white shadow-lg transition hover:bg-[#0a2530] focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/80"
            >
              <svg
                className="h-3.5 w-3.5"
                viewBox="0 0 24 24"
                fill="currentColor"
                aria-hidden
              >
                <rect
                  x="6"
                  y="6"
                  width="12"
                  height="12"
                  rx="2.5"
                  ry="2.5"
                />
              </svg>
            </button>
          ) : (
            <button
              type="submit"
              disabled={!value.trim()}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 flex h-9 w-9 items-center justify-center rounded-xl bg-cyan-600 text-white shadow-lg shadow-cyan-600/20 transition hover:bg-cyan-500 disabled:pointer-events-none disabled:opacity-40"
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
          )}
        </div>
      </form>
    </div>
  );
}
