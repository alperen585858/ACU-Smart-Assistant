import type { SessionMeta } from "../types";

type Props = {
  sessions: SessionMeta[];
  activeId: string | null;
  isOpen: boolean;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onDelete: (id: string, e: React.MouseEvent) => void;
};

export default function SessionSidebar({
  sessions,
  activeId,
  isOpen,
  onSelect,
  onNewChat,
  onDelete,
}: Props) {
  return (
    <div
      id="chat-history-panel"
      className={`flex shrink-0 flex-col overflow-hidden border-[#0b2e3b]/10 bg-white/45 transition-[width,opacity] duration-200 ease-out ${
        isOpen
          ? "w-[min(13.5rem,42vw)] border-r opacity-100"
          : "w-0 border-0 opacity-0"
      }`}
    >
      <div className="flex h-full w-[min(13.5rem,42vw)] flex-col">
        <button
          type="button"
          onClick={onNewChat}
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
          New chat
        </button>

        <nav className="mt-2 flex-1 overflow-y-auto px-2 pb-3">
          <p className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-wide text-[#0b2e3b]/45">
            History (server)
          </p>
          <ul className="space-y-0.5">
            {sessions.map((s) => (
              <li key={s.id}>
                <div
                  role="button"
                  tabIndex={0}
                  onClick={() => onSelect(s.id)}
                  onKeyDown={(ev) => {
                    if (ev.key === "Enter" || ev.key === " ") {
                      ev.preventDefault();
                      void onSelect(s.id);
                    }
                  }}
                  className={`group flex w-full items-center gap-0.5 rounded-lg px-2 py-1.5 text-left text-xs transition sm:text-[13px] ${
                    s.id === activeId
                      ? "bg-cyan-600/15 font-medium text-[#0b2e3b] ring-1 ring-cyan-600/25"
                      : "text-[#0b2e3b]/80 hover:bg-white/50"
                  }`}
                >
                  <span
                    className="min-w-0 flex-1 truncate"
                    title={s.title}
                  >
                    {s.title}
                  </span>
                  <button
                    type="button"
                    onClick={(ev) => void onDelete(s.id, ev)}
                    className="shrink-0 rounded-md p-0.5 text-[#0b2e3b]/35 opacity-0 transition hover:bg-red-500/15 hover:text-red-700 group-hover:opacity-100"
                    aria-label="Delete conversation"
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
  );
}
