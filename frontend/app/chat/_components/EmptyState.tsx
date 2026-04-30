type Props = {
  onSuggestionClick: (text: string) => void;
};

const SUGGESTIONS = [
  "Tell me about Acibadem University",
  "What programs and faculties are there?",
  "Short overview of the campus",
];

export default function EmptyState({ onSuggestionClick }: Props) {
  return (
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
          Hi — how can I help you today?
        </p>
        <p className="mt-2 max-w-sm text-sm text-[#0b2e3b]/65">
          Ask me anything about Acibadem University (programs, campus,
          admissions, and more).
        </p>
      </div>

      <div className="flex flex-wrap justify-center gap-2">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => onSuggestionClick(suggestion)}
            className="rounded-full border border-white/40 bg-white/40 px-4 py-2 text-sm text-[#0b2e3b]/80 transition-colors hover:bg-white/65"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}
