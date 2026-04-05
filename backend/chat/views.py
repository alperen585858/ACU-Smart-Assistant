import json
import os
import re
import socket
import urllib.error
import urllib.request
import uuid

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from pgvector.django import CosineDistance

from .models import ChatMessage, ChatSession
from core.embeddings import embed_query
from core.models import DocumentChunk

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")
RAG_MAX_CHARS = 2000
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "10"))
# Cosine distance (lower = closer). 0.55 was too strict—correct pages often sat at 0.56–0.65.
RAG_MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", "0.62"))
# When no chunk passes the threshold, still send the closest K (model must follow “only if in CONTEXT”).
RAG_RELAX_ON_EMPTY = os.environ.get("RAG_RELAX_ON_EMPTY", "true").lower() in (
    "1",
    "true",
    "yes",
)
# Merge chunks whose text contains query keywords (helps Turkish/synonym misses vs BGE-en only).
RAG_KEYWORD_BOOST = os.environ.get("RAG_KEYWORD_BOOST", "true").lower() in (
    "1",
    "true",
    "yes",
)
RAG_SNIPPET_CHARS = max(400, int(os.environ.get("RAG_SNIPPET_CHARS", "900")))

# Ollama settings
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
# Defaults must fit RAG system prompt + Context (~2k chars) + history; tiny ctx/predict
# causes truncation so the model never sees the full address and may refuse or hallucinate.
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "384"))
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
OLLAMA_HTTP_TIMEOUT = int(os.environ.get("OLLAMA_HTTP_TIMEOUT", "240"))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.15"))
OLLAMA_TOP_P = float(os.environ.get("OLLAMA_TOP_P", "0.85"))
# Long threads exceed Ollama num_ctx; the model then drops early tokens (system+RAG) and
# falls back to generic "as an AI / 2023" disclaimers despite crawled Context.
CHAT_HISTORY_MAX_MESSAGES = max(1, int(os.environ.get("CHAT_HISTORY_MAX_MESSAGES", "12")))
CHAT_MESSAGE_MAX_CHARS = max(200, int(os.environ.get("CHAT_MESSAGE_MAX_CHARS", "900")))

# Claude API settings
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_BASE = (
    "You are the official website assistant for Acıbadem Mehmet Ali Aydınlar University (ACU). "
    "LANGUAGE: English only. Never use Turkish unless the user pasted Turkish inside their message. "
    "Be direct and factual. "
    "GROUNDING: Use ONLY the Context block below when it is present. Quote or paraphrase it; "
    "do not invent addresses, policies, disclaimers, or refusals. "
    "Official campus or unit postal addresses, phone numbers, and emails printed in Context are "
    "public university contact data—state them when the user asks; do not refuse as \"private\". "
    "Do not mention OpenAI, Anthropic, Microsoft, training data, or content policies. "
    "If Context is missing or does not contain the answer, reply exactly: "
    "\"I don't have that information in the crawled pages. Try running a data refresh or rephrase your question.\" "
    "If the question is out of scope for the website content, say so in one sentence."
)

# When RAG hits, put crawled text in the *last user* turn (not only at end of system).
# Many local models truncate from the start of the prompt; system+RAG at the top was being dropped
# while long chat history remained, so the model answered from pretrained “Microsoft / 2023” habits.
SYSTEM_RAG_USER_WRAPPER = (
    "You are the official Acıbadem Mehmet Ali Aydınlar University (ACU) website assistant. "
    "LANGUAGE: English only unless the user pasted Turkish inside ===QUESTION=== below. "
    "The user message has ===CONTEXT=== (excerpts from crawled acibadem.edu.tr) and ===QUESTION===. "
    "Rules: (1) Every factual claim must be supported by ===CONTEXT===; do not invent or guess. "
    "(2) Do not add rankings, statistics, dates, program names, fees, or partner universities unless "
    "those exact facts appear in ===CONTEXT===. Do not fill gaps from memory. "
    "(3) Default: 2–4 short sentences. If ===QUESTION=== asks for a broad overview, 'everything', "
    "'all you know', or similar, and ===CONTEXT=== is non-empty, summarize supported facts from "
    "===CONTEXT=== in up to 8 short sentences—still only facts present in those excerpts. "
    "(4) Never say you are from Microsoft/OpenAI/Anthropic; never mention training data, browsing "
    "the live web, or a knowledge cutoff year. "
    "(5) If ===CONTEXT=== is non-empty and the user asks generally what the university is or wants "
    "a wide summary, you MUST answer from ===CONTEXT=== (do not refuse). "
    "(6) Use the refusal ONLY when ===CONTEXT=== is empty OR the user asks for one specific fact "
    "that does not appear anywhere in ===CONTEXT===. Refusal text (exact, nothing before or after): "
    "\"I don't have that information in the crawled pages. Try running a data refresh or rephrase your question.\" "
    "(7) Never append topics (e.g. scholarships) to the refusal unless that topic is in ===QUESTION===."
)
RAG_USER_BUBBLE_MAX_CHARS = max(2000, int(os.environ.get("RAG_USER_BUBBLE_MAX_CHARS", "4500")))

RAG_META_REASON_SKIPPED_SMALLTALK = "skipped_smalltalk_no_rag"

_BROAD_OVERVIEW_QUESTION_RE = re.compile(
    r"\b(everything|all\s+(you\s+)?know|write\s+all|tell\s+me\s+(everything|all)|overview|"
    r"summarize|summary|what\s+do\s+you\s+know|genel\s+bilgi|hepsini)\b",
    re.IGNORECASE,
)

# Greetings / thanks — do not run RAG (irrelevant chunks force bogus "data refresh" answers).
SYSTEM_SMALLTALK = (
    "You are the official website assistant for Acıbadem Mehmet Ali Aydınlar University (ACU). "
    "LANGUAGE: English only unless the user wrote in Turkish. "
    "The user sent a short greeting or courtesy, not a factual question about the university. "
    "Reply with 1–2 brief, friendly sentences: greet back and offer help with ACU topics "
    "(programs, admissions, campus, contact). "
    "Do not mention crawled pages, data refresh, scholarships, or training data unless they asked. "
    "Do not mention OpenAI, Anthropic, or Microsoft. "
    "Do not repeat or mimic wording from earlier assistant messages in this chat."
)

_SMALLTALK_RE = re.compile(
    r"^[\s!?.`,]*("
    r"(hi|hello|hey|yo|hiya)(\s+(there|everyone|all|guys|team))?"
    r"|merhaba(\s+nasilsin|\s+nasılsın)?|selam(\s+aleykum)?|\bsa\b|\bslm\b"
    r"|good\s+(morning|afternoon|evening|night)(\s+there)?"
    r"|how\s+are\s+you(\s+doing)?|what'?s\s+up|\bwassup\b|you\s+ok\?"
    r"|thanks?(\s+a\s+lot)?|thank\s+you(\s+so\s+much)?|\bthx\b|\bty\b"
    r"|teşekkürler?|tesekkurler?"
    r"|\bok\b|okay|tamam|\bbye\b|goodbye|see\s+you|güle\s+güle"
    r")[\s!?.`,]*$",
    re.IGNORECASE,
)

# If these appear, user likely wants facts — keep RAG on (do not use smalltalk path).
# Note: do not use bare "what"/"how" here — they appear in "what's up" / "how are you".
_SMALLTALK_EXCLUDE_RE = re.compile(
    r"\b(when|where|which|who|why|burs|scholar|tuition|fee|program|programs?|adres|address|"
    r"başvuru|basvuru|kampüs|kampus|contact|iletişim|ücret|cret|apply|application|"
    r"deadline|calendar|course|exam|graduate|undergraduate)\b|"
    r"\bwhat\s+(is|are|was|were|does|did|do|can|should|about|if)\b|"
    r"\bhow\s+(do|can|i|to|much|many|long|about|apply|register)\b",
    re.IGNORECASE,
)


def _should_skip_rag_for_smalltalk(user_plain: str) -> bool:
    t = (user_plain or "").strip()
    if not t or len(t) > 160:
        return False
    compact = " ".join(t.split())
    if not _SMALLTALK_RE.match(compact):
        return False
    if _SMALLTALK_EXCLUDE_RE.search(compact):
        return False
    return True


_RAG_KEYWORD_STOP = frozenset(
    """
    what when where which who how why the and for with about from that this have does did you your
    are was were please dont not tell can could would should university universite universitesi
    acibadem acıbadem mehmet ali aydinlar aydınlar tell lie know please more some any very just
    like into than then them they their there here hakkında nedir nelerdir nasıl hangi şey
    """.split()
)


def _rag_keywords_from_query(text: str, max_terms: int = 5) -> list[str]:
    raw = (text or "").lower()
    words = re.findall(r"[a-zA-ZğüşıöçĞÜŞİÖÇ]{4,}", raw)
    out: list[str] = []
    for w in words:
        if w in _RAG_KEYWORD_STOP:
            continue
        if w not in out:
            out.append(w)
        if len(out) >= max_terms:
            break
    return out


def _compose_rag_search_query(
    current_message: str, prior_user_messages: list[str]
) -> str:
    """
    BGE embedding model is English-first; short Turkish questions often miss relevant chunks.
    Merge recent user turns and anchor with the university name when absent.
    """
    prior = [t.strip() for t in prior_user_messages if t.strip()][-2:]
    cur = (current_message or "").strip()
    merged = "\n".join(prior + [cur]) if (prior or cur) else ""
    if not merged:
        return ""
    if not re.search(
        r"acibadem|\bacu\b|mehmet\s+ali|açıbadem|aydınlar",
        merged,
        re.IGNORECASE,
    ):
        merged = f"{merged}\nAcıbadem Mehmet Ali Aydınlar University (ACU)"
    # BGE is English-heavy; Turkish location/contact questions often retrieve wrong pages
    # (runtime logs: "universite adresi" ranked scholarship pages; English "address" hit ACUTAB).
    if re.search(
        r"adres|address|konum|location|postal|tam\s*adres|kamp[uü]s|campus|"
        r"\bnerede\b|where\s+is|iletişim|contact\b|ulaşım|how\s+to\s+get",
        merged,
        re.IGNORECASE,
    ):
        merged = f"{merged}\npostal address campus location contact Istanbul Kerem Aydinlar"
    return merged


def _search_pages_with_meta(query: str) -> tuple[str, list[dict], bool, bool]:
    """
    Returns (context_text, sources, used_relaxed_fallback, embedding_ok).
    """
    query = (query or "").strip()
    if not query:
        return "", [], False, True

    query_vector = embed_query(query)
    if not query_vector:
        return "", [], False, False

    base_qs = (
        DocumentChunk.objects.annotate(distance=CosineDistance("embedding", query_vector))
        .order_by("distance")
    )

    has_rows = DocumentChunk.objects.exists()
    used_relaxed = False
    if RAG_RELAX_ON_EMPTY and has_rows:
        vector_chunks = list(base_qs.filter(distance__lte=RAG_MAX_DISTANCE)[:RAG_TOP_K])
        if not vector_chunks:
            vector_chunks = list(base_qs[:RAG_TOP_K])
            used_relaxed = bool(vector_chunks)
    else:
        vector_chunks = list(base_qs.filter(distance__lte=RAG_MAX_DISTANCE)[:RAG_TOP_K])

    # (chunk, distance) — keyword rows use a nominal distance for ordering/display only
    ranked: list[tuple] = []
    seen_pk: set[int] = set()
    for ch in vector_chunks:
        pk = ch.pk
        if pk not in seen_pk:
            seen_pk.add(pk)
            ranked.append((ch, float(ch.distance)))

    if RAG_KEYWORD_BOOST and has_rows:
        for term in _rag_keywords_from_query(query):
            for ch in DocumentChunk.objects.filter(content__icontains=term)[:3]:
                pk = ch.pk
                if pk not in seen_pk:
                    seen_pk.add(pk)
                    ranked.append((ch, 0.75))

    context_parts: list[str] = []
    sources: list[dict] = []
    total = 0
    seen_urls: set[str] = set()
    seen_chunk_ids: set[int] = set()

    for chunk, dist_val in ranked:
        if chunk.source_url in seen_urls and total > int(RAG_MAX_CHARS * 0.7):
            continue
        seen_urls.add(chunk.source_url)

        snippet = (chunk.content or "")[:RAG_SNIPPET_CHARS]
        if total + len(snippet) > RAG_MAX_CHARS:
            break
        title = chunk.page_title or chunk.source_url
        context_parts.append(f"[{title}]\n{snippet}")
        total += len(snippet)
        cid = getattr(chunk, "pk", None)
        if cid is not None and cid not in seen_chunk_ids:
            seen_chunk_ids.add(cid)
            sources.append(
                {
                    "url": chunk.source_url,
                    "title": (title or "")[:200],
                    "cosine_distance": round(float(dist_val), 4),
                }
            )

    return "\n\n".join(context_parts), sources, used_relaxed, True


def _wrap_user_with_rag_context(context: str, user_plain: str) -> str:
    footer = (
        "\n===END_QUESTION===\n"
        "Answer in English using only facts from ===CONTEXT===. "
        "If the question is broad or asks for 'everything', summarize what ===CONTEXT=== actually says. "
        "Use the system-message refusal only when ===CONTEXT=== is empty or the specific fact is missing. "
        "If you refuse, output the refusal sentence alone with no added clauses."
    )
    body = (
        f"===CONTEXT===\n{context.strip()}\n===QUESTION===\n{user_plain.strip()}{footer}"
    )
    if len(body) > RAG_USER_BUBBLE_MAX_CHARS:
        # Keep question + footer; trim context from the end
        qpart = f"\n===QUESTION===\n{user_plain.strip()}{footer}"
        overhead = len("===CONTEXT===\n\n...(truncated)...\n")
        room = RAG_USER_BUBBLE_MAX_CHARS - overhead - len(qpart)
        ctx = context.strip()[: max(500, room)]
        body = f"===CONTEXT===\n{ctx}\n...(truncated)...{qpart}"
    return body


def _attach_llm_visibility_meta(meta: dict, user_llm: str, context_char_count: int) -> dict:
    """Prove to the client whether crawled text was actually placed in the prompt."""
    meta["indexed_chunks_in_db"] = DocumentChunk.objects.count()
    meta["context_chars_sent"] = context_char_count
    meta["llm_user_turn_chars"] = len(user_llm)
    meta["context_block_in_llm"] = bool(
        context_char_count > 0 and "===CONTEXT===" in user_llm
    )
    return meta


def _prepare_chat_prompts(rag_query: str, user_plain: str) -> tuple[str, str, dict]:
    """
    Build (system_message, user_message_for_llm, rag_meta).
    When retrieval succeeds, crawled excerpts live in the user turn so they stay near the end
    of the prompt and survive context-window truncation better than system-only RAG.
    """
    user_plain = (user_plain or "").strip()

    if _should_skip_rag_for_smalltalk(user_plain):
        meta = {
            "embedding_ok": True,
            "chunks_used": 0,
            "relaxed_retrieval": False,
            "sources": [],
            "rag_query_preview": "",
            "reason": RAG_META_REASON_SKIPPED_SMALLTALK,
        }
        user_llm = _trim_message_for_llm(user_plain)
        _attach_llm_visibility_meta(meta, user_llm, 0)
        return SYSTEM_SMALLTALK, user_llm, meta

    context, sources, relaxed, emb_ok = _search_pages_with_meta(rag_query)
    meta: dict = {
        "embedding_ok": emb_ok,
        "chunks_used": len(sources),
        "relaxed_retrieval": relaxed,
        "sources": sources,
        "rag_query_preview": rag_query[:400],
    }

    if not emb_ok:
        system = (
            f"{SYSTEM_BASE}\n\nThe question could not be embedded (model error). "
            "Use the exact fallback sentence from the rules."
        )
        user_llm = _trim_message_for_llm(user_plain)
        _attach_llm_visibility_meta(meta, user_llm, 0)
        return system, user_llm, meta

    if context:
        system = SYSTEM_RAG_USER_WRAPPER
        if relaxed:
            system += (
                "\n\nNote: Strict vector match was weak; ===CONTEXT=== is still the closest crawl text. "
                "For broad or overview questions, summarize facts it contains. "
                "For one narrow fact, cite it only if that fact clearly appears in ===CONTEXT===."
            )
        if _BROAD_OVERVIEW_QUESTION_RE.search(user_plain):
            system += (
                "\n\nHIGH PRIORITY: The user asked for a broad summary. ===CONTEXT=== is non-empty — "
                "answer by summarizing concrete facts stated in those excerpts (names, units, places). "
                "Do not refuse unless the excerpts are truly empty of relevant facts."
            )
        user_llm = _wrap_user_with_rag_context(context, user_plain)
        _attach_llm_visibility_meta(meta, user_llm, len(context))
        return system, user_llm, meta

    meta["reason"] = "no_matching_chunks"
    system = (
        f"{SYSTEM_BASE}\n\n"
        "No Context was retrieved for this question. Follow the fallback rule; "
        "do not guess from general knowledge."
    )
    user_llm = _trim_message_for_llm(user_plain)
    _attach_llm_visibility_meta(meta, user_llm, 0)
    return system, user_llm, meta


def _trim_last_user_for_llm(content: str) -> str:
    """Do not apply 900-char cap to RAG-wrapped user bubbles (would delete context)."""
    if "===CONTEXT===" in content:
        if len(content) > RAG_USER_BUBBLE_MAX_CHARS + 500:
            return content[: RAG_USER_BUBBLE_MAX_CHARS + 499] + "…"
        return content
    return _trim_message_for_llm(content)


def _rag_query_from_request_body(body: dict) -> str:
    """Derive retrieval text from JSON body (message and/or messages[])."""
    user_msg = (body.get("message") or "").strip()
    raw_history = body.get("messages")
    if isinstance(raw_history, list) and raw_history:
        users: list[str] = []
        for item in raw_history:
            if isinstance(item, dict) and item.get("role") == "user":
                c = (item.get("content") or "").strip()
                if c:
                    users.append(c)
        if users:
            return _compose_rag_search_query(users[-1], users[:-1])
    if user_msg:
        return _compose_rag_search_query(user_msg, [])
    return ""


def _trim_message_for_llm(text: str, max_chars: int = CHAT_MESSAGE_MAX_CHARS) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def _parse_client_id(raw: str | None):
    if not raw:
        return None
    try:
        return uuid.UUID(str(raw))
    except (ValueError, TypeError):
        return None


def _call_claude(messages: list) -> tuple[str | None, str | None]:
    system_text = ""
    api_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            api_messages.append({"role": m["role"], "content": m["content"]})

    payload = json.dumps({
        "model": CLAUDE_MODEL,
        "max_tokens": 256,
        "system": system_text,
        "messages": api_messages,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        return None, f"Claude API hatası: {detail}"
    except urllib.error.URLError as e:
        return None, f"Claude API bağlantı hatası: {e.reason}"

    content_blocks = data.get("content", [])
    reply = ""
    for block in content_blocks:
        if block.get("type") == "text":
            reply += block.get("text", "")
    reply = reply.strip()
    if not reply:
        return None, "Claude boş yanıt döndü"
    return reply, None


def _call_ollama(ollama_messages: list) -> tuple[str | None, str | None]:
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": ollama_messages,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "num_predict": OLLAMA_NUM_PREDICT,
                "num_ctx": OLLAMA_NUM_CTX,
                "temperature": OLLAMA_TEMPERATURE,
                "top_p": OLLAMA_TOP_P,
            },
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        return None, detail or e.reason
    except urllib.error.URLError as e:
        err = str(e.reason)
        if isinstance(e.reason, TimeoutError) or "timed out" in err.lower():
            return None, (
                "Ollama yanıtı zaman aşımına uğradı. Docker RAM artırın, "
                "OLLAMA_NUM_PREDICT azaltın veya modeli küçültün."
            )
        return None, err
    except socket.timeout:
        return None, "Ollama zaman aşımı (socket). Sunucu veya model çok yavaş."

    reply = (data.get("message") or {}).get("content", "").strip()
    if not reply:
        return None, "Empty model response"
    return reply, None


def _call_llm(messages: list) -> tuple[str | None, str | None]:
    if LLM_BACKEND == "claude" and ANTHROPIC_API_KEY:
        return _call_claude(messages)
    return _call_ollama(messages)


@csrf_exempt
@require_GET
def list_sessions(request):
    cid = _parse_client_id(
        request.GET.get("client_id") or request.headers.get("X-Client-Id")
    )
    if cid is None:
        return JsonResponse({"error": "client_id gerekli (UUID)"}, status=400)
    sessions = ChatSession.objects.filter(client_id=cid)[:100]
    return JsonResponse(
        {
            "sessions": [
                {
                    "id": str(s.id),
                    "title": s.title,
                    "updated_at": s.updated_at.isoformat(),
                }
                for s in sessions
            ]
        }
    )


@csrf_exempt
@require_http_methods(["GET", "DELETE"])
def session_detail(request, pk):
    cid = _parse_client_id(
        request.GET.get("client_id") or request.headers.get("X-Client-Id")
    )
    if cid is None:
        return JsonResponse({"error": "client_id gerekli (query, UUID)"}, status=400)

    session = get_object_or_404(ChatSession, pk=pk, client_id=cid)

    if request.method == "DELETE":
        session.delete()
        return JsonResponse({"ok": True})

    msgs = [
        {
            "id": str(m.id),
            "role": m.role,
            "content": m.content,
            "timestamp": m.created_at.isoformat(),
        }
        for m in session.messages.all()
    ]
    return JsonResponse({"session_id": str(session.id), "title": session.title, "messages": msgs})


@csrf_exempt
@require_http_methods(["POST"])
def chat_completion(request):
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    client_uuid = _parse_client_id(body.get("client_id"))

    if client_uuid is not None:
        return _chat_with_db(request, body, client_uuid)

    raw_history = body.get("messages")
    user_msg = (body.get("message") or "").strip()
    rag_q = _rag_query_from_request_body(body)

    if isinstance(raw_history, list) and len(raw_history) > 0:
        parsed: list[dict] = []
        for item in raw_history[-CHAT_HISTORY_MAX_MESSAGES:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = (item.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                parsed.append({"role": role, "content": _trim_message_for_llm(content)})

        last_user_idx: int | None = None
        plain_for_rag = user_msg
        for i in range(len(parsed) - 1, -1, -1):
            if parsed[i]["role"] == "user":
                last_user_idx = i
                plain_for_rag = parsed[i]["content"]
                break

        if last_user_idx is None and not user_msg:
            return JsonResponse(
                {"error": "messages must include at least one user turn"},
                status=400,
            )
        if last_user_idx is None:
            plain_for_rag = user_msg

        system_text, user_llm, rag_meta = _prepare_chat_prompts(rag_q, plain_for_rag)
        ollama_messages: list = [{"role": "system", "content": system_text}]
        if rag_meta.get("reason") == RAG_META_REASON_SKIPPED_SMALLTALK:
            ollama_messages.append(
                {"role": "user", "content": _trim_last_user_for_llm(user_llm)}
            )
        elif last_user_idx is None:
            for m in parsed:
                ollama_messages.append(m)
            ollama_messages.append(
                {"role": "user", "content": _trim_last_user_for_llm(user_llm)}
            )
        else:
            for i, m in enumerate(parsed):
                if i == last_user_idx:
                    ollama_messages.append(
                        {
                            "role": "user",
                            "content": _trim_last_user_for_llm(user_llm),
                        }
                    )
                else:
                    ollama_messages.append(m)
        if len(ollama_messages) < 2:
            return JsonResponse(
                {"error": "messages must include at least one user/assistant turn"},
                status=400,
            )
    else:
        message = (body.get("message") or "").strip()
        if not message:
            return JsonResponse({"error": "message or messages is required"}, status=400)
        system_text, user_llm, rag_meta = _prepare_chat_prompts(rag_q, message)
        ollama_messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": _trim_last_user_for_llm(user_llm)},
        ]

    reply_text, err = _call_llm(ollama_messages)
    if err:
        status = 504 if "zaman aşımı" in err.lower() or "timeout" in err.lower() else 502
        return JsonResponse({"error": err, "rag": rag_meta}, status=status)
    return JsonResponse({"reply": reply_text, "rag": rag_meta})


def _chat_with_db(request, body: dict, client_uuid: uuid.UUID) -> JsonResponse:
    message = (body.get("message") or "").strip()
    if not message:
        return JsonResponse({"error": "message is required"}, status=400)

    session_id_raw = body.get("session_id")
    if session_id_raw:
        try:
            sid = uuid.UUID(str(session_id_raw))
        except (ValueError, TypeError):
            return JsonResponse({"error": "session_id geçersiz UUID"}, status=400)
        session = get_object_or_404(ChatSession, pk=sid, client_id=client_uuid)
    else:
        session = ChatSession.objects.create(client_id=client_uuid, title="Yeni sohbet")

    prior = list(session.messages.all())
    prior_user_texts = [m.content for m in prior if m.role == "user"]
    rag_q = _compose_rag_search_query(message, prior_user_texts)
    system_text, user_llm, rag_meta = _prepare_chat_prompts(rag_q, message)
    ollama_messages: list = [{"role": "system", "content": system_text}]
    # Greetings / thanks: do not inject prior turns — models echo previous bad RAG refusals.
    if rag_meta.get("reason") != RAG_META_REASON_SKIPPED_SMALLTALK:
        prior_window = prior[-CHAT_HISTORY_MAX_MESSAGES:]
        for m in prior_window:
            if m.role in ("user", "assistant") and m.content.strip():
                ollama_messages.append(
                    {
                        "role": m.role,
                        "content": _trim_message_for_llm(m.content),
                    }
                )
    ollama_messages.append(
        {"role": "user", "content": _trim_last_user_for_llm(user_llm)}
    )

    user_row = ChatMessage.objects.create(
        session=session, role="user", content=message
    )
    title_changed = False
    if session.title == "Yeni sohbet" and len(prior) == 0:
        session.title = message[:197] + ("…" if len(message) > 200 else "")
        session.save(update_fields=["title"])
        title_changed = True

    reply_text, err = _call_llm(ollama_messages)
    if err:
        user_row.delete()
        if title_changed:
            session.title = "Yeni sohbet"
            session.save(update_fields=["title"])
        status = 504 if "zaman aşımı" in err.lower() or "timeout" in err.lower() else 502
        return JsonResponse(
            {"error": err, "session_id": str(session.id), "rag": rag_meta},
            status=status,
        )

    ChatMessage.objects.create(session=session, role="assistant", content=reply_text)
    session.save()

    return JsonResponse(
        {
            "reply": reply_text,
            "session_id": str(session.id),
            "title": session.title,
            "rag": rag_meta,
        }
    )
