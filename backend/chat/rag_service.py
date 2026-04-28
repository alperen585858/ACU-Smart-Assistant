import os
import re

from core.rag_keywords import (
    RAG_ACADEMIC_OBS_INTENT_RE,
    RAG_BROAD_FEE_LIST_INTENT_RE,
    RAG_DEPT_OR_FACULTY_INTENT_RE,
    RAG_FACULTY_ROSTER_INTENT_RE,
    RAG_FEE_TUITION_INTENT_RE,
    RAG_LEADERSHIP_INTENT_RE,
    RAG_LOCATION_CONTACT_INTENT_RE,
    RAG_STEM_OR_ENGINEERING_INTENT_RE,
    extract_target_entity_key,
    faculty_roster_path_filter,
    fee_tuition_intent,
    graduate_or_postgrad_admissions_intent,
    international_admissions_default_undergraduate_only,
    international_application_requirements_page_intent,
    international_student_apply_intent,
)
from core.rag_retrieval import search_document_chunks

from .message_utils import trim_message_for_llm

RAG_USER_BUBBLE_MAX_CHARS = max(2000, int(os.environ.get("RAG_USER_BUBBLE_MAX_CHARS", "14000")))
RAG_META_REASON_SKIPPED_SMALLTALK = "skipped_smalltalk_no_rag"

SYSTEM_BASE = (
    "You are the official website assistant for Acıbadem Mehmet Ali Aydınlar University (ACU). "
    "LANGUAGE: English only. Never use Turkish unless the user pasted Turkish inside their message. "
    "Be factual but approachable: in English, sound like a helpful human assistant, not a terse FAQ bot. "
    "When you successfully answer in English (not when using the exact refusal below), end with one short "
    "warm line offering further help, e.g. ‘Is there anything else I can help you with?’ or "
    "’Let me know if you’d like details on programs, admissions, or campus life.’ "
    "GROUNDING: Use ONLY the Context block below when it is present. Quote or paraphrase it; "
    "do not invent addresses, policies, disclaimers, or refusals. "
    "Official campus or unit postal addresses, phone numbers, and emails printed in Context are "
    "public university contact data—state them when the user asks; do not refuse as \"private\". "
    "Do not mention OpenAI, Anthropic, Microsoft, training data, or content policies. "
    "If Context is missing or does not contain the answer, reply exactly (no extra sentences before or after): "
    "\"I don’t have that information in the crawled pages. Try running a data refresh or rephrase your question.\" "
    "If the question is out of scope for the website content, say so in one warm sentence and offer to help "
    "with ACU topics instead."
)

SYSTEM_RAG_USER_WRAPPER = (
    "You are the official Acıbadem Mehmet Ali Aydınlar University (ACU) website assistant. "
    "LANGUAGE: Default to polished, natural English for English questions—that is the usual case. "
    "Use the same warm, human assistant tone in English as you would in Turkish: never dry or telegraphic. "
    "If the user’s question is clearly in Turkish only, answer in polite Turkish at the same warmth level. "
    "The user message contains crawled website excerpts, then the user’s question (marked with internal "
    "delimiters for your eyes only). "
    "CRITICAL OUTPUT RULES: Never write ===CONTEXT===, ===QUESTION===, ===END_QUESTION===, or any similar "
    "delimiter text in your answer. Never begin with ‘According to ===CONTEXT===’ or ‘Based on ===CONTEXT===’. "
    "STYLE (English and Turkish): Warm, courteous, professional—like helpful front-desk staff. "
    "For ENGLISH replies: you may open with one short friendly line when it fits, e.g. ‘Happy to help!’, "
    "’Sure—here’s what I found on the website.’, ‘I’d be glad to help with that.’, then give the facts. "
    "Use fluent complete sentences; avoid robotic or blunt FAQ tone. "
    "For fee or program lists you may use short bullet points only when many items must be shown; "
    "otherwise prefer a short paragraph. Do not paste raw bracket titles like [Page title | Site]; "
    "say the source naturally (e.g. ‘on the tuition page’). "
    "CLOSING: For every normal factual answer (not the exact refusal in rule 6), end with exactly one short "
    "follow-up offer in the SAME language as your answer. "
    "ENGLISH closings (vary these): ‘Is there anything else I can help you with?’, "
    "’Let me know if you’d like more detail on programs, fees, or campus life.’, "
    "’Feel free to ask if you have other questions about ACU.’ "
    "Keep the closing brief. Do not add it after the exact refusal sentence in rule 6. "
    "Rules: (1) Every factual claim must be supported by the excerpts; do not invent or guess. "
    "(2) Do not invent rankings, statistics, dates, fees, or partner universities. "
    "You MAY name faculties, schools, departments, and programs exactly as written in the excerpts. "
    "Do not fill gaps from memory. "
    "(3) BREVITY: Keep answers short and to the point—2–4 sentences for simple questions, "
    "up to 6 sentences for broad overviews. Never pad with filler or repeat information. "
    "Include all essential facts from the excerpts but cut unnecessary elaboration. "
    "Bullet points only when listing 3+ items; otherwise use a concise paragraph. "
    "(4) Never say you are from Microsoft/OpenAI/Anthropic; never mention training data, browsing "
    "the live web, or a knowledge cutoff year. "
    "(5) If excerpts are non-empty and the user asks generally what the university is or wants a wide summary, "
    "you MUST answer from those excerpts (do not refuse). "
    "(6) Use the refusal ONLY when excerpts are empty OR the user asks for one specific fact that does not "
    "appear in the excerpts. Refusal text (exact—output this sentence alone with no greeting and no closing offer): "
    "\"I don’t have that information in the crawled pages. Try running a data refresh or rephrase your question.\" "
    "(7) Never append topics (e.g. scholarships) to the refusal unless the user asked about them. "
    "(8) If the user asks for departments, faculties, schools, or programs, list every such unit named in "
    "the excerpts; if incomplete, list what is present and note the crawl may be partial—never claim nothing "
    "is listed when the excerpts name any unit. "
    "(9) If the user names a specific field (e.g. Computer Engineering) and that phrase or a clear Turkish "
    "equivalent appears in the excerpts, affirm it from the text; do not claim it is missing unless those "
    "strings truly do not appear. "
    "(10) Tuition or fees: if the user asked about one specific department or program, do not cite another "
    "unit’s fees as if they were for that unit. If they asked for all programs, the full fee list, or general "
    "university tuition, you may summarize every program and amount that appears in the excerpts. "
    "If the named program is not mentioned in the excerpts, say it is not listed in this retrieved text. "
    "If the program is named but a fee figure is not next to it in the excerpts, do not invent an amount; "
    "you may use the rule 6 refusal for the missing number only, not to deny that the program exists. "
    "Do not borrow numbers from unrelated passages. "
    "(11) Admissions / apply (especially for international or foreign students): do not treat tuition, fee, or "
    "price pages as proof that someone may apply unless the same excerpt clearly states who may apply. "
    "If excerpts include an eligibility or “who can apply” list, summarize those points; do not tell the user "
    "to navigate the website. Do not write vague padding such as “criteria are not specified in detail” or "
    "implied eligibility. If excerpts do not clearly answer, use rule 6 (exact refusal) with no extra sentences "
    "or fake offers. "
    "(12) When excerpts list required diplomas/exams and minimum scores in a table, report those rows with the "
    "same names and numbers shown; do not collapse into vague wording or say details are missing if they are in "
    "the text; do not add graduate or master’s rules unless the excerpt is about that level. "
    "(13) If the user did not ask for graduate studies (master’s, PhD, graduate program), do not require that "
    "applicants already hold a specific bachelor’s or university degree in a subject; international "
    "undergraduate admission is normally about high school completion and listed entrance exams, unless the "
    "excerpts clearly refer to the level the user asked for."
)

SYSTEM_SMALLTALK = (
    "You are the official website assistant for Acıbadem Mehmet Ali Aydınlar University (ACU). "
    "LANGUAGE: If the user wrote in English, reply entirely in warm, natural English. "
    "Only use Turkish if their message is clearly in Turkish. "
    "The user sent a short greeting or courtesy, not a factual question about the university. "
    "Reply with 2–3 brief, warm sentences: greet back, say you’re happy to help with ACU, and invite a next step. "
    "In ENGLISH include a gentle offer such as ‘What would you like to know?’ or "
    "’How can I help you today—programs, admissions, campus, or contact?’ "
    "In Turkish you might use ‘Ne konuda yardımcı olayım?’ or similar. "
    "Do not mention crawled pages, data refresh, scholarships, or training data unless they asked. "
    "Do not mention OpenAI, Anthropic, or Microsoft. "
    "Do not repeat or mimic wording from earlier assistant messages in this chat."
)

_BROAD_OVERVIEW_QUESTION_RE = re.compile(
    r"\b(everything|all\s+(you\s+)?know|write\s+all|tell\s+me\s+(everything|all)|overview|"
    r"summarize|summary|what\s+do\s+you\s+know|genel\s+bilgi|hepsini)\b",
    re.IGNORECASE,
)
_SMALLTALK_RE = re.compile(
    r"^[\s!?.`,;'\"]*("
    r"(hi|hello|hey|yo|hiya|sup|hola|howdy)(\s+(there|everyone|all|guys|team|baby|bro|dude|man))?"
    r"|merhaba(\s+nasilsin|\s+nasılsın)?|selam(\s+aleykum)?|\bsa\b|\bslm\b|\bnaber\b"
    r"|good\s+(morning|afternoon|evening|night)(\s+there)?"
    r"|how\s+are\s+you(\s+doing)?|what'?s\s+up|\bwassup\b|you\s+ok\?|what'?s\s+good"
    r"|yo\s+what'?s\s+good"
    r"|thanks?(\s+a\s+lot)?|thank\s+you(\s+so\s+much)?|\bthx\b|\bty\b"
    r"|teşekkürler?|tesekkurler?|sağ\s*ol|sagol"
    r"|\bok\b|okay|tamam|\bbye\b|goodbye|see\s+you|güle\s+güle|\bbb\b"
    r"|nice|cool|great|awesome|perfect|alright"
    r")[\s!?.`,;'\"]*$",
    re.IGNORECASE,
)
_SMALLTALK_EXCLUDE_RE = re.compile(
    r"\b(when|where|which|who|why|burs|scholar|tuition|fee|program|programs?|adres|address|"
    r"başvuru|basvuru|kampüs|kampus|contact|iletişim|ücret|cret|apply|application|"
    r"deadline|calendar|course|exam|graduate|undergraduate)\b|"
    r"\bwhat\s+(is|are|was|were|does|did|do|can|should|about|if)\b|"
    r"\bhow\s+(do|can|i|to|much|many|long|about|apply|register)\b",
    re.IGNORECASE,
)


_OFFTOPIC_RE = re.compile(
    r"\b(weather|forecast|hava\s*durumu|recipe|tarif|joke|fıkra|espri|"
    r"movie|film|music|müzik|song|şarkı|game|oyun|football|futbol|"
    r"basketball|basketbol|bitcoin|crypto|kripto|stock|borsa|"
    r"diet|diyet|horoscope|burç|netflix|spotify|instagram|tiktok|"
    r"who\s+is\s+elon|who\s+is\s+trump|who\s+is\s+biden|"
    r"write\s+me\s+a\s+(poem|story|code|essay)|"
    r"translate|çevir|what\s+time|saat\s+kaç)\b",
    re.IGNORECASE,
)
_OFFTOPIC_EXCLUDE_RE = re.compile(
    r"\b(acu|acıbadem|acibadem|university|üniversite|campus|kampüs|"
    r"faculty|fakülte|program|department|bölüm|student|öğrenci|"
    r"admission|kayıt|tuition|ücret|scholarship|burs)\b",
    re.IGNORECASE,
)

SYSTEM_OFFTOPIC = (
    "You are the official website assistant for Acıbadem Mehmet Ali Aydınlar University (ACU). "
    "The user asked something unrelated to the university. "
    "Politely say you can only help with ACU-related topics and offer to assist with "
    "programs, admissions, campus life, or contact information. "
    "Keep it to 1-2 warm sentences. Match the user's language (English or Turkish)."
)
RAG_META_REASON_SKIPPED_OFFTOPIC = "skipped_offtopic_no_rag"


def _should_skip_rag_for_offtopic(user_plain: str) -> bool:
    t = (user_plain or "").strip()
    if not t or len(t) > 300:
        return False
    if _OFFTOPIC_EXCLUDE_RE.search(t):
        return False
    return bool(_OFFTOPIC_RE.search(t))


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


def _looks_english_only(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    # If Turkish-specific characters exist, treat as non-English.
    if re.search(r"[çğıöşüÇĞİÖŞÜ]", t):
        return False
    # If contains at least one ASCII letter and no Turkish chars, prefer English.
    return bool(re.search(r"[A-Za-z]", t))


def compose_rag_search_query(current_message: str, prior_user_messages: list[str]) -> str:
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
    # Apply at most one intent-specific keyword boost to avoid diluting embeddings.
    if RAG_LOCATION_CONTACT_INTENT_RE.search(merged):
        merged = (
            f"{merged}\n"
            "postal address campus location contact communication transportation directions "
            "how to get Istanbul Kerem Aydinlar"
        )
    elif RAG_ACADEMIC_OBS_INTENT_RE.search(merged):
        merged = (
            f"{merged}\n"
            "OBS Bologna course catalog curriculum syllabus ECTS program outcomes"
        )
    elif international_student_apply_intent(merged):
        merged = (
            f"{merged}\n"
            "Acıbadem University international students admission application requirements "
            "how to apply international office yabancı öğrenci başvuru yükseköğretim kabul"
        )
        if not graduate_or_postgrad_admissions_intent(merged):
            merged = (
                f"{merged}\n"
                "undergraduate first-cycle international admission (pre-university high school diploma and "
                "entrance exams), not master PhD or graduate school unless the user asked for those"
            )
        if fee_tuition_intent(merged):
            merged = (
                f"{merged}\n"
                "international student tuition and fees (only if the user also asked about cost or price)"
            )
        if international_application_requirements_page_intent(merged):
            merged = (
                f"{merged}\n"
                "Acıbadem undergraduate international Application Requirements table "
                "required diploma exam TR-YÖS SAT GCE ACT AP IB School of Medicine English Turkish program scores"
            )
    elif RAG_FEE_TUITION_INTENT_RE.search(merged):
        # Default: all faculties / programs (crawl may hold one page with many program rows).
        # Narrow to one department only when a known dept phrase matches AND user did not ask for
        # "all fees" (tüm / all programs / etc.).
        path_seg = faculty_roster_path_filter(merged)
        broad_fees = bool(RAG_BROAD_FEE_LIST_INTENT_RE.search(merged)) or path_seg is None
        if broad_fees:
            merged = (
                f"{merged}\n"
                "Acıbadem University all undergraduate graduate programs tuition and fees "
                "all faculties schools Medicine Health Sciences Engineering Law Pharmacy "
                "Dentistry vocational associate degree master doctoral öğrenim ücreti "
                "fee schedule list annual"
            )
        else:
            merged = (
                f"{merged}\n"
                "Acıbadem University tuition and fees admissions "
                "undergraduate graduate program fee öğrenim ücreti annual cost"
            )
            if path_seg:
                label = path_seg.replace("-", " ")
                merged = f"{merged}\n{label} program tuition fee"
                if path_seg == "faculty-of-health-sciences":
                    merged = (
                        f"{merged}\n"
                        "Acıbadem Faculty of Health Sciences Sağlık Bilimleri "
                        "Physiotherapy Nursing Nutrition Dietetics Healthcare Management — "
                        "not School of Medicine not associate vocational MYO not Engineering"
                    )
                if path_seg == "faculty-of-medicine":
                    merged = (
                        f"{merged}\n"
                        "Acıbadem Faculty of Medicine Tıp Fakültesi lisans 6 year undergraduate "
                        "MD hekimlik program tuition fee not Medical Education master yüksek lisans "
                        "not pedagogy not vocational techniques MYO"
                    )
                if path_seg == "computer-programming":
                    merged = (
                        f"{merged}\n"
                        "Computer Programming associate degree ön lisans not Computer Engineering "
                        "not engineering faculty tuition row disambiguation"
                    )
    elif RAG_FACULTY_ROSTER_INTENT_RE.search(merged) and (
        RAG_STEM_OR_ENGINEERING_INTENT_RE.search(merged)
        or faculty_roster_path_filter(merged)
    ):
        merged = (
            f"{merged}\n"
            "department academic staff page faculty members professors "
            "instructors by name and title"
        )
    elif RAG_LEADERSHIP_INTENT_RE.search(merged):
        merged = (
            f"{merged}\n"
            "Acıbadem faculty deans rector leadership organization schools management board"
        )
    elif RAG_STEM_OR_ENGINEERING_INTENT_RE.search(cur):
        merged = (
            f"{merged}\nComputer Engineering undergraduate "
            "faculty engineering program degree"
        )
    elif RAG_DEPT_OR_FACULTY_INTENT_RE.search(cur):
        merged = (
            f"{merged}\nfaculty school department Fakülte "
            "programs schools list"
        )
    return merged


def rag_query_from_request_body(body: dict) -> str:
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
            return compose_rag_search_query(users[-1], users[:-1])
    if user_msg:
        return compose_rag_search_query(user_msg, [])
    return ""


def _search_pages_with_meta(composed_query: str, raw_user_query: str = "") -> tuple[str, list[dict], bool, bool]:
    return search_document_chunks(composed_query, raw_user_query or None)


def _wrap_user_with_rag_context(context: str, user_plain: str) -> str:
    footer = (
        "\n===END_QUESTION===\n"
        "Now write your reply to the user. Use only facts from the excerpts above. "
        "If the question is in English, write in warm, natural English (not terse): optional one-line opener, "
        "then facts, then one short English offer to help further. Same idea in Turkish for Turkish questions. "
        "Do not repeat the words ===CONTEXT=== or ===QUESTION=== in your reply. "
        "If the question is broad, summarize what the excerpts actually state. "
        "Follow the system message for opening and closing (except when using the exact refusal). "
        "Use the system refusal sentence only when excerpts are empty or the specific fact is missing."
    )
    body = f"===CONTEXT===\n{context.strip()}\n===QUESTION===\n{user_plain.strip()}{footer}"
    if len(body) > RAG_USER_BUBBLE_MAX_CHARS:
        qpart = f"\n===QUESTION===\n{user_plain.strip()}{footer}"
        overhead = len("===CONTEXT===\n\n...(truncated)...\n")
        room = RAG_USER_BUBBLE_MAX_CHARS - overhead - len(qpart)
        ctx = context.strip()[: max(500, room)]
        body = f"===CONTEXT===\n{ctx}\n...(truncated)...{qpart}"
    return body


def _attach_llm_visibility_meta(meta: dict, user_llm: str, context_char_count: int) -> dict:
    meta["context_chars_sent"] = context_char_count
    meta["llm_user_turn_chars"] = len(user_llm)
    meta["context_block_in_llm"] = bool(context_char_count > 0 and "===CONTEXT===" in user_llm)
    return meta


def prepare_chat_prompts(rag_query: str, user_plain: str) -> tuple[str, str, dict]:
    user_plain = (user_plain or "").strip()
    force_english = _looks_english_only(user_plain)

    # Layer 1: Smalltalk — skip RAG entirely
    if _should_skip_rag_for_smalltalk(user_plain):
        meta = {
            "embedding_ok": True,
            "chunks_used": 0,
            "relaxed_retrieval": False,
            "sources": [],
            "rag_query_preview": "",
            "reason": RAG_META_REASON_SKIPPED_SMALLTALK,
        }
        user_llm = trim_message_for_llm(user_plain)
        _attach_llm_visibility_meta(meta, user_llm, 0)
        return SYSTEM_SMALLTALK, user_llm, meta

    # Layer 2: Off-topic — skip RAG, polite redirect
    if _should_skip_rag_for_offtopic(user_plain):
        meta = {
            "embedding_ok": True,
            "chunks_used": 0,
            "relaxed_retrieval": False,
            "sources": [],
            "rag_query_preview": "",
            "reason": RAG_META_REASON_SKIPPED_OFFTOPIC,
        }
        user_llm = trim_message_for_llm(user_plain)
        _attach_llm_visibility_meta(meta, user_llm, 0)
        return SYSTEM_OFFTOPIC, user_llm, meta

    # Layer 3: Real question — go to RAG
    context, sources, relaxed, emb_ok = _search_pages_with_meta(rag_query, user_plain)
    meta: dict = {
        "embedding_ok": emb_ok,
        "chunks_used": len(sources),
        "relaxed_retrieval": relaxed,
        "sources": sources,
        "rag_query_preview": rag_query[:400],
    }
    if not emb_ok and not context:
        system = (
            f"{SYSTEM_BASE}\n\nThe question could not be embedded (model error). "
            "Use the exact fallback sentence from the rules."
        )
        if force_english:
            system += " LANGUAGE OVERRIDE: Reply strictly in English."
        user_llm = trim_message_for_llm(user_plain)
        _attach_llm_visibility_meta(meta, user_llm, 0)
        return system, user_llm, meta

    if context:
        system = SYSTEM_RAG_USER_WRAPPER
        if force_english:
            system += "\n\nLANGUAGE OVERRIDE: The user's message is English. Reply strictly in English."
        if relaxed:
            system += (
                "\n\nNote: Strict vector match was weak; the excerpts are still the closest crawl text. "
                "For broad or overview questions, summarize facts they contain. "
                "For one narrow fact, state it only if it clearly appears in the excerpts."
            )
        if _BROAD_OVERVIEW_QUESTION_RE.search(user_plain):
            system += (
                "\n\nHIGH PRIORITY: The user asked for a broad summary. The excerpts are non-empty — "
                "answer by summarizing concrete facts stated there (names, units, places). "
                "Do not refuse unless the excerpts are truly empty of relevant facts."
            )
        if RAG_DEPT_OR_FACULTY_INTENT_RE.search(user_plain) and not fee_tuition_intent(
            user_plain
        ):
            system += (
                "\n\nThe user asks for departments/faculties/schools. Extract and list every distinct "
                "faculty, school, or department name that appears in the excerpts; if incomplete, "
                "still list what is present."
            )
        if RAG_LEADERSHIP_INTENT_RE.search(user_plain):
            system += (
                "\n\nThe user asks about deans, rector, or top academic leadership. "
                "Name any deans, rectors, or vice-rectors explicitly stated in the excerpts; "
                "if an excerpt only gives a dean’s office, email, or phone, say that — do not claim "
                "the university does not mention leadership unless the excerpts truly have no such terms."
            )
        if RAG_STEM_OR_ENGINEERING_INTENT_RE.search(user_plain):
            system += (
                "\n\nThe user asked about a degree, engineering discipline, or program. If the excerpts "
                "name that program in English or Turkish, answer from those lines only; say it is not "
                "mentioned only if neither the English nor Turkish program name appears there."
            )
        if RAG_FACULTY_ROSTER_INTENT_RE.search(user_plain) and "faculty listing" in (
            context or ""
        ):
            system += (
                "\n\nMANDATORY (faculty list in Context): Answer with concrete names and titles from the "
                "faculty listing block first. Do not reply with only generic offers such as ‘What would you "
                "like to know?’ or a vague invitation to ask about the university. Do not ask follow-up "
                "questions before you have given the list. If the listing block contains any person names, "
                "you must enumerate them; only refuse if there are truly zero names in the Context. "
                "Include only people who appear in that department’s staff list text. Do not add faculty "
                "from Psychology, Biomedical, Medicine, or other units unless the staff list text itself "
                "names them as part of the same department roster—do not invent or import names from memory."
            )
        if international_student_apply_intent(user_plain):
            system += (
                "\n\nADMISSIONS (international / foreign): The user is asking about applying, requirements, or "
                "eligibility for international applicants. (a) If the excerpts list who can or cannot apply, "
                "base your answer on that list in short bullet points. (b) Do not use tuition/fee pages as the "
                "main evidence for who may apply. (c) Do not tell the user to browse the site, open menus, or "
                "“go to the International section” — state only what the excerpts contain. (d) Do not name "
                "schools, exams, or countries unless they appear in the excerpts. (e) If the excerpts do not list "
                "requirements, use rule 6 only—no filler."
            )
        if international_application_requirements_page_intent(user_plain):
            system += (
                "\n\nINTERNATIONAL UNDERGRADUATE APPLICATION REQUIREMENTS: If the Context includes a “REQUIRED "
                "DIPLOMA/EXAM” (or similar) list or table with School of Medicine / English / Turkish program "
                "minimum scores, you MUST summarize using those exact exam or diploma types and the scores shown — "
                "do not collapse into “various diplomas” or “no specific details.” Do not add graduate, associate, "
                "or language-test rules unless they appear in the same Context block. You may add one sentence that "
                "meeting the table does not guarantee admission if that sentence appears in the excerpt."
            )
        if international_admissions_default_undergraduate_only(user_plain):
            system += (
                "\n\nUNDERGRADUATE-FOCUS (the user did not ask for master’s, PhD, or graduate program): Answer as "
                "if they are asking about first university entry (typical: high school / secondary completion + "
                "entrance exam requirements). **Do not** state that the applicant must already have a **bachelor’s or "
                "university degree in a subject** (e.g. Chemistry) — that is usually graduate admissions language. "
                "If a Context line looks like a graduate or postgraduate requirement, do not use it for this question "
                "unless the user explicitly asked about that level. Prefer the undergraduate / international table "
                "excerpts only."
            )
        if fee_tuition_intent(user_plain) and faculty_roster_path_filter(
            user_plain
        ) == "faculty-of-medicine":
            system += (
                "\n\nMEDICINE TUITION (English “medicine” = Tıp Fakültesi / undergraduate MD, unless the user said "
                "“master’s” or “graduate” explicitly): Do NOT equate with: (1) “Medical Education” or similar master’s "
                "programs, (2) Tıp Eğitimi yüksek lisans, (3) any health pedagogy or medical-technique associate "
                "diploma, (4) “Medical … Techniques” lines. If the only fee rows name such programs, the undergraduate "
                "Faculty of Medicine / hekimlik amount is not in the excerpt—say that and the rule 6 refusal for the exact "
                "MD fee only; do not quote 3,500 USD (or any figure) from a Medical Education or technique program as "
                "if it were the standard “medicine” (MD) tuition. Only use an amount if the excerpt clearly labels "
                "Faculty of Medicine, Tıp, hekimlik, or the six-year MD / Tıp lisans next to that price."
            )
        elif fee_tuition_intent(user_plain) and faculty_roster_path_filter(user_plain):
            system += (
                "\n\nFEE + NAMED DEPARTMENT: Search the Context for that program in English and Turkish. "
                "If a price appears on the same line or table row, report it. If the program name appears but "
                "no amount is in the excerpt, say the specific fee is not in this retrieved text; do not claim "
                "the university does not offer the program. Use rule 6 only when the Context has no relevant fee text."
            )
        target_entity = extract_target_entity_key(user_plain)
        if target_entity:
            system += (
                "\n\nENTITY GROUNDING (MANDATORY): Answer only from excerpts that explicitly mention "
                "the same target entity as the user question. If excerpts mention only related but different "
                "entities, do not transfer names, titles, or numbers across them. If the target entity is named "
                "but the exact fact is missing, state that this specific detail is not present in the retrieved text."
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
    if force_english:
        system += " LANGUAGE OVERRIDE: Reply strictly in English."
    user_llm = trim_message_for_llm(user_plain)
    _attach_llm_visibility_meta(meta, user_llm, 0)
    return system, user_llm, meta
