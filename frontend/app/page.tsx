import Link from "next/link";

export default function Home() {
  return (
    <div className="relative flex min-h-screen flex-col overflow-hidden bg-[#04032a] font-sans">
      {/* Background layers */}
      <div className="pointer-events-none absolute inset-0" aria-hidden>
        <div className="absolute -left-[18%] -top-[32%] h-[70%] w-[48%] rounded-full bg-cyan-400/10 blur-[120px]" />
        <div className="absolute -bottom-[32%] -right-[12%] h-[75%] w-[55%] rounded-full bg-blue-500/20 blur-[140px]" />
        <div
          className="absolute inset-0 opacity-[0.06]"
          style={{
            backgroundImage: `linear-gradient(rgba(0,255,255,.11) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(0,255,255,.11) 1px, transparent 1px)`,
            backgroundSize: "120px 120px",
          }}
        />
        <div
          className="absolute inset-0 opacity-[0.2]"
          style={{
            backgroundImage: `radial-gradient(1000px 420px at 18% 24%, rgba(65, 234, 255, 0.18), transparent 70%),
                              radial-gradient(900px 360px at 82% 84%, rgba(65, 120, 255, 0.25), transparent 72%)`,
          }}
        />
      </div>

      <div className="relative z-10 flex min-h-screen flex-col">
        {/* Top bar */}
        <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-6 sm:px-8">
          <span className="font-display text-lg font-semibold tracking-[0.02em] text-cyan-100">
            ACU Smart Assistant
          </span>
          <Link
            href="/chat"
            className="rounded-full border border-cyan-300/35 bg-cyan-300/12 px-5 py-2 text-sm font-semibold text-cyan-100 transition-all hover:bg-cyan-300/25 hover:text-white"
          >
            Go to chat
          </Link>
        </header>

        {/* Hero */}
        <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col items-center justify-center px-6 pb-24 pt-8 text-center sm:px-8">
          <p className="font-display mb-5 text-3xl font-bold uppercase tracking-[0.22em] text-cyan-300">
            ACU SMART ASSISTANT
          </p>
          <h1 className="font-display max-w-4xl text-5xl font-extrabold leading-[1.08] tracking-tight text-white sm:text-6xl md:text-7xl">
            Ask your questions.
            <br />
            <span className="bg-gradient-to-r from-cyan-300 via-cyan-200 to-blue-300 bg-clip-text text-transparent">
              Get answers.
            </span>
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-cyan-50/85">
            Instant answers, code examples, and ideas with our text-based chat
            assistant. Try it free.
          </p>

          {/* CTA */}
          <Link
            href="/chat"
            className="group mt-10 inline-flex items-center gap-3 rounded-full bg-cyan-300 px-10 py-4 text-base font-bold uppercase tracking-[0.14em] text-[#091240] shadow-[0_0_0_1px_rgba(56,245,255,0.45),0_10px_30px_-14px_rgba(56,245,255,0.95)] transition-all hover:-translate-y-0.5 hover:bg-cyan-200 active:scale-[0.98]"
          >
            Start chatting
            <svg
              className="h-5 w-5 transition-transform group-hover:translate-x-0.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M13 7l5 5m0 0l-5 5m5-5H6"
              />
            </svg>
          </Link>

          {/* Features */}
          <ul className="mt-14 flex flex-wrap items-center justify-center gap-x-8 gap-y-3 text-sm text-cyan-100/75">
            <li className="flex items-center gap-2">
              <span className="flex h-1.5 w-1.5 rounded-full bg-cyan-300" />
              Instant answers
            </li>
            <li className="flex items-center gap-2">
              <span className="flex h-1.5 w-1.5 rounded-full bg-cyan-200" />
              Code & explanation
            </li>
            <li className="flex items-center gap-2">
              <span className="flex h-1.5 w-1.5 rounded-full bg-blue-300" />
              Free to use
            </li>
          </ul>
        </main>

        {/* Footer */}
        <footer className="relative z-10 border-t border-cyan-300/15 px-6 py-4 sm:px-8">
          <p className="text-center text-xs text-cyan-100/50">
            ACU Smart Assistant · Chat assistant
          </p>
        </footer>
      </div>
    </div>
  );
}
