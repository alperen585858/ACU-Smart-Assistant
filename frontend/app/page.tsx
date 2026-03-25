import Link from "next/link";

export default function Home() {
  return (
    <div className="relative min-h-screen overflow-hidden bg-[#04032a] font-sans">
      {/* Campus background */}
      <div className="pointer-events-none fixed inset-0 -z-10" aria-hidden>
        <div
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: `url("/kampus.png.webp")` }}
        />
        <div className="absolute inset-0 bg-[#04032a]/70" />
      </div>

      {/* Top mini header */}
      <header className="relative z-10 mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-5 sm:px-6">
        <div className="flex items-center gap-3">
          <a
            href="/chat"
            className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white/15 ring-1 ring-white/15 transition hover:bg-white/25"
            aria-label="Go to chat"
          >
            <img src="/logo.svg" alt="ACU" className="h-8 w-auto" />
          </a>
          <div>
            <div className="text-sm font-semibold text-white/90">
              ACU Smart Assistant
            </div>
            <div className="text-xs text-white/50">Campus knowledge chat</div>
          </div>
        </div>

        <Link
          href="/chat"
          className="hidden rounded-full border border-white/15 bg-white/5 px-4 py-2 text-sm font-medium text-white/80 transition hover:bg-white/10 sm:inline-flex"
        >
          Go to chat
        </Link>
      </header>

      {/* Center card */}
      <main className="relative z-10 mx-auto flex min-h-[calc(100vh-80px)] items-center justify-center px-4 py-10 sm:px-6">
        <div className="w-full max-w-3xl overflow-hidden rounded-3xl border border-white/30 bg-white/75 backdrop-blur-xl shadow-[0_20px_60px_rgba(0,0,0,0.35)]">
          <div className="px-6 py-6 sm:px-10 sm:py-10">
            <div className="flex flex-col items-center text-center">
              <p className="text-base font-semibold tracking-[0.20em] text-[#0b2e3b]/60">
                ACU SMART ASSISTANT
              </p>
              <h1 className="mt-2 text-3xl font-extrabold tracking-tight text-[#0b2e3b] sm:text-4xl">
                Ask your questions.
              </h1>
              <p className="mt-3 max-w-2xl text-sm leading-relaxed text-[#0b2e3b]/65 sm:text-base">
                Instant answers powered by your campus content. Try it in chat
                right now.
              </p>

              <div className="mt-7 flex w-full flex-col items-center gap-3 sm:flex-row sm:justify-center">
                <Link
                  href="/chat"
                  className="inline-flex items-center justify-center gap-2 rounded-2xl bg-cyan-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-cyan-600/20 transition hover:bg-cyan-500"
                >
                  Start chatting
                  <svg
                    className="h-4 w-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2.2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M13 7l5 5m0 0l-5 5m5-5H6"
                    />
                  </svg>
                </Link>
              </div>

              {/* Features */}
              <ul className="mt-8 grid w-full grid-cols-1 gap-3 sm:grid-cols-3">
                <li className="rounded-2xl border border-white/45 bg-white/50 px-4 py-3">
                  <div className="text-sm font-semibold text-[#0b2e3b]">
                    Instant answers
                  </div>
                  <div className="mt-1 text-xs text-[#0b2e3b]/60">
                    Get responses quickly.
                  </div>
                </li>
                <li className="rounded-2xl border border-white/45 bg-white/50 px-4 py-3">
                  <div className="text-sm font-semibold text-[#0b2e3b]">
                    Campus-focused
                  </div>
                  <div className="mt-1 text-xs text-[#0b2e3b]/60">
                    Built around university info.
                  </div>
                </li>
                <li className="rounded-2xl border border-white/45 bg-white/50 px-4 py-3">
                  <div className="text-sm font-semibold text-[#0b2e3b]">
                    Easy to use
                  </div>
                  <div className="mt-1 text-xs text-[#0b2e3b]/60">
                    Simple chat interface.
                  </div>
                </li>
              </ul>
            </div>
          </div>

          <div className="flex items-center justify-center border-t border-white/35 px-6 py-4 text-xs text-[#0b2e3b]/55">
            ACU Smart Assistant · Chat assistant
          </div>
        </div>
      </main>
    </div>
  );
}
