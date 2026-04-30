type Props = {
  elapsedSec: number;
  onStop: () => void;
};

export default function TypingIndicator({ elapsedSec, onStop }: Props) {
  return (
    <div className="flex justify-start animate-fade-in">
      <div className="flex max-w-full items-center gap-3 rounded-3xl border border-white/35 bg-gradient-to-r from-white/70 to-white/55 py-3 pl-5 pr-3 shadow-sm ring-1 ring-cyan-500/10 backdrop-blur-sm sm:gap-4 sm:pr-4">
        <div className="flex min-w-0 flex-1 items-center gap-3 sm:gap-4">
          <div className="flex items-center gap-1.5" aria-hidden>
            <span className="h-2 w-2 animate-bounce rounded-full bg-gradient-to-br from-cyan-500 to-cyan-600 [animation-delay:-0.32s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-gradient-to-br from-cyan-500 to-cyan-600 [animation-delay:-0.16s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-gradient-to-br from-cyan-500 to-cyan-600" />
          </div>
          <div className="flex min-w-0 flex-1 flex-col leading-none">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-[#0b2e3b]/40">
              Waiting for response
            </span>
            <span className="mt-1 font-mono text-lg font-semibold tabular-nums tracking-tight text-cyan-800">
              {elapsedSec.toFixed(1)}
              <span className="ml-0.5 text-sm font-medium text-[#0b2e3b]/45">
                s
              </span>
            </span>
          </div>
        </div>
        <button
          type="button"
          onClick={onStop}
          title="Stop"
          aria-label="Stop generating response"
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-[#0b2e3b]/15 bg-[#0b2e3b]/90 text-white shadow-md transition hover:bg-[#0b2e3b] focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/80"
        >
          <svg
            className="h-4 w-4"
            viewBox="0 0 24 24"
            fill="currentColor"
            aria-hidden
          >
            <rect x="6" y="6" width="12" height="12" rx="2.5" ry="2.5" />
          </svg>
        </button>
      </div>
    </div>
  );
}
