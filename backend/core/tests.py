"""Tests for core app: chunking, URL normalization, scraper helpers.

These tests are designed to run without Django/PostgreSQL dependencies.
Run with: python -m unittest core.tests -v
"""

import unittest
from urllib.parse import urldefrag, urlparse

from core.chunking import chunk_text, chunks_for_embedding
from core.html_extract import extract_title_and_text, extract_title_text_and_embedding_units
from core.rag_query_expand import (
    snippet_around_phrase,
    whois_name_from_queries,
    whois_name_in_content,
    whois_vector_variants,
)


ALLOWED_NETLOCS = frozenset({"acibadem.edu.tr"})


def is_english_path(path):
    normalized_path = (path or "/").rstrip("/").lower()
    return normalized_path == "/en" or normalized_path.startswith("/en/")


def normalize_url(url, english_only=True):
    url, _frag = urldefrag(url.strip())
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url.lstrip("/")
        parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ""
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in ALLOWED_NETLOCS:
        return ""
    path = parsed.path or "/"
    if english_only and not is_english_path(path):
        return ""
    return f"{parsed.scheme}://{host}{path}" + (
        f"?{parsed.query}" if parsed.query else ""
    )


def same_site(url):
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host in ALLOWED_NETLOCS
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class ChunkTextTests(unittest.TestCase):
    """Tests for the text chunking algorithm."""

    def test_empty_string_returns_empty(self):
        self.assertEqual(chunk_text(""), [])

    def test_none_returns_empty(self):
        self.assertEqual(chunk_text(None), [])

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(chunk_text("   \n\t  "), [])

    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world", chunk_size=700, chunk_overlap=120)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Hello world")

    def test_exact_chunk_size(self):
        chunks = chunk_text("a" * 700, chunk_size=700, chunk_overlap=120)
        self.assertEqual(len(chunks), 1)

    def test_overlap_creates_multiple_chunks(self):
        text = "word " * 200  # 1000 chars
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
        self.assertGreater(len(chunks), 1)

    def test_overlap_must_be_less_than_chunk_size(self):
        with self.assertRaises(ValueError):
            chunk_text("test", chunk_size=100, chunk_overlap=100)

    def test_whitespace_is_normalized(self):
        chunks = chunk_text("hello   world\n\nfoo   bar", chunk_size=700, chunk_overlap=120)
        self.assertEqual(chunks[0], "hello world foo bar")

    def test_long_text_no_empty_chunks(self):
        chunks = chunk_text("test " * 500, chunk_size=200, chunk_overlap=50)
        for chunk in chunks:
            self.assertTrue(len(chunk) > 0)

    def test_chunks_cover_all_content(self):
        text = "abcdefghij" * 10  # 100 chars
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
        combined = "".join(chunks)
        # All original chars should appear at least once
        for char in set(text):
            self.assertIn(char, combined)


class NormalizeUrlTests(unittest.TestCase):
    """Tests for URL normalization."""

    def test_basic_english_url(self):
        result = normalize_url("https://www.acibadem.edu.tr/en")
        self.assertEqual(result, "https://acibadem.edu.tr/en")

    def test_www_stripped(self):
        result = normalize_url("https://www.acibadem.edu.tr/en/about")
        self.assertNotIn("www.", result)

    def test_fragment_removed(self):
        result = normalize_url("https://acibadem.edu.tr/en/page#section")
        self.assertNotIn("#", result)

    def test_non_english_path_filtered(self):
        result = normalize_url("https://acibadem.edu.tr/hakkimizda", english_only=True)
        self.assertEqual(result, "")

    def test_non_english_path_allowed(self):
        result = normalize_url("https://acibadem.edu.tr/hakkimizda", english_only=False)
        self.assertNotEqual(result, "")

    def test_external_domain_rejected(self):
        result = normalize_url("https://google.com/en")
        self.assertEqual(result, "")

    def test_no_scheme_adds_https(self):
        result = normalize_url("acibadem.edu.tr/en")
        self.assertTrue(result.startswith("https://"))

    def test_ftp_rejected(self):
        result = normalize_url("ftp://acibadem.edu.tr/en")
        self.assertEqual(result, "")

    def test_empty_string(self):
        self.assertEqual(normalize_url(""), "")

    def test_query_params_preserved(self):
        result = normalize_url("https://acibadem.edu.tr/en/search?q=test")
        self.assertIn("?q=test", result)


class IsEnglishPathTests(unittest.TestCase):
    """Tests for English path detection."""

    def test_en_root(self):
        self.assertTrue(is_english_path("/en"))

    def test_en_subpath(self):
        self.assertTrue(is_english_path("/en/about"))

    def test_en_trailing_slash(self):
        self.assertTrue(is_english_path("/en/"))

    def test_turkish_path(self):
        self.assertFalse(is_english_path("/hakkimizda"))

    def test_root_path(self):
        self.assertFalse(is_english_path("/"))

    def test_similar_but_not_en(self):
        self.assertFalse(is_english_path("/energy"))

    def test_case_insensitive(self):
        self.assertTrue(is_english_path("/EN/about"))


class SameSiteTests(unittest.TestCase):
    """Tests for same-site checking."""

    def test_acibadem_is_same_site(self):
        self.assertTrue(same_site("https://acibadem.edu.tr/en"))

    def test_www_acibadem_is_same_site(self):
        self.assertTrue(same_site("https://www.acibadem.edu.tr/en"))

    def test_external_is_not_same_site(self):
        self.assertFalse(same_site("https://google.com"))

    def test_invalid_url(self):
        self.assertFalse(same_site("not a url"))

    def test_empty_string(self):
        self.assertFalse(same_site(""))


class ExtractTitleTextAndEmbeddingUnitsTests(unittest.TestCase):
    """DOM-derived embedding units (staff-style tables and lists)."""

    def test_table_rows_become_units(self):
        html = """<html><body><main><table>
<tr><th>Name</th><th>Role</th></tr>
<tr><td>Ata Akin</td><td>Dean</td></tr>
<tr><td>Jane Doe</td><td>Head of Department</td></tr>
</table></main></body></html>"""
        title, text, units = extract_title_text_and_embedding_units(html)
        self.assertGreaterEqual(len(units), 2)
        self.assertTrue(any("Ata Akin" in u and "Dean" in u for u in units))
        self.assertTrue(any("Jane Doe" in u for u in units))
        self.assertIn("Ata Akin", text)

    def test_top_level_list_items(self):
        html = """<html><body><main><ul>
<li>Alpha Role — Person One</li>
<li>Beta Role — Person Two</li>
</ul></main></body></html>"""
        _title, _text, units = extract_title_text_and_embedding_units(html)
        self.assertEqual(len(units), 2)
        self.assertIn("Person One", units[0])


class ChunksForEmbeddingTests(unittest.TestCase):
    """Entity-aware chunking vs. hierarchical fallback."""

    def test_structural_units_one_chunk_per_short_row(self):
        content = "Staff\n\nAta Akin\nDean\n\nJane Doe\nHead"
        units = ["Ata Akin\nDean", "Jane Doe\nHead"]
        chunks = chunks_for_embedding(content, units, chunk_size=700, chunk_overlap=120)
        self.assertEqual(len(chunks), 2)
        self.assertIn("Ata Akin", chunks[0])

    def test_fallback_when_units_too_sparse(self):
        long_intro = "Lorem ipsum " * 200
        content = long_intro + "\n\nOnly tiny list tail."
        units = ["nav item"]
        chunks = chunks_for_embedding(content, units, chunk_size=200, chunk_overlap=40)
        self.assertGreater(len(chunks), 1)

    def test_oversized_unit_splits_without_full_whitespace_collapse(self):
        line = "Title: " + ("word " * 80)
        units = [line]
        chunks = chunks_for_embedding("x", units, chunk_size=120, chunk_overlap=20)
        self.assertGreaterEqual(len(chunks), 1)


class ExtractTitleAndTextTests(unittest.TestCase):
    """Tests for HTML content extraction."""

    def test_basic_html(self):
        html = "<html><head><title>Test Page</title></head><body><p>Hello world</p></body></html>"
        title, text = extract_title_and_text(html)
        self.assertEqual(title, "Test Page")
        self.assertIn("Hello world", text)

    def test_scripts_removed(self):
        html = "<html><body><p>Content</p><script>alert('xss')</script></body></html>"
        _, text = extract_title_and_text(html)
        self.assertNotIn("alert", text)
        self.assertIn("Content", text)

    def test_styles_removed(self):
        html = "<html><body><style>.x{color:red}</style><p>Visible</p></body></html>"
        _, text = extract_title_and_text(html)
        self.assertNotIn("color", text)
        self.assertIn("Visible", text)

    def test_empty_html(self):
        title, text = extract_title_and_text("")
        self.assertEqual(title, "")

    def test_no_title(self):
        html = "<html><body><p>No title here</p></body></html>"
        title, text = extract_title_and_text(html)
        self.assertEqual(title, "")
        self.assertIn("No title here", text)

    def test_content_truncated(self):
        html = f"<html><body><p>{'x' * 10000}</p></body></html>"
        _, text = extract_title_and_text(html)
        self.assertLessEqual(len(text), 5000)

    def test_prefers_main_tag(self):
        html = "<html><body><header>Header</header><main><p>Main content</p></main><footer>Footer</footer></body></html>"
        _, text = extract_title_and_text(html)
        self.assertIn("Main content", text)

    def test_noscript_removed(self):
        html = "<html><body><noscript>Enable JS</noscript><p>Real</p></body></html>"
        _, text = extract_title_and_text(html)
        self.assertNotIn("Enable JS", text)
        self.assertIn("Real", text)


class NormalizeObsUrlTests(unittest.TestCase):
    """Tests for OBS Bologna URL normalization (no browser required)."""

    def test_showpac_absolute(self):
        from core.obs_bologna_scraper import normalize_obs_url

        base = "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en"
        u = normalize_obs_url(
            "https://obs.acibadem.edu.tr/oibs/bologna/showPac.aspx?code=1",
            base,
        )
        self.assertTrue(u)
        self.assertIn("showPac", u)

    def test_relative_bologna_path(self):
        from core.obs_bologna_scraper import normalize_obs_url

        base = "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en"
        u = normalize_obs_url("/oibs/bologna/other.aspx?x=1", base)
        self.assertTrue(u)
        self.assertIn("/oibs/bologna/", u)

    def test_rejects_non_obs_host(self):
        from core.obs_bologna_scraper import normalize_obs_url

        base = "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en"
        self.assertEqual(
            "", normalize_obs_url("https://example.com/oibs/bologna/x", base)
        )

    def test_rejects_javascript(self):
        from core.obs_bologna_scraper import normalize_obs_url

        base = "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en"
        self.assertEqual("", normalize_obs_url("javascript:void(0)", base))

    def test_strips_fragment(self):
        from core.obs_bologna_scraper import normalize_obs_url

        base = "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en"
        u = normalize_obs_url(
            "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en#foo",
            base,
        )
        self.assertNotIn("#", u)


class TestRagQueryExpand(unittest.TestCase):
    """Asymmetric person-name query expansion (no Django)."""

    def test_whois_english_name(self):
        self.assertEqual(
            whois_name_from_queries("Who is Ahmet Bulut?", None),
            "Ahmet Bulut",
        )
        self.assertEqual(
            whois_name_from_queries("Please tell me who is Ahmet Bulut", ""),
            "Ahmet Bulut",
        )

    def test_kimdir_turkish(self):
        self.assertEqual(
            whois_name_from_queries("Ahmet Bulut kimdir", None),
            "Ahmet Bulut",
        )
        self.assertEqual(
            whois_name_from_queries("kimdir Ahmet Bulut", None),
            "Ahmet Bulut",
        )

    def test_not_role_only_query(self):
        self.assertIsNone(
            whois_name_from_queries("head of computer engineering at ACU", None)
        )

    def test_whois_uses_raw_when_composed_has_acu_suffix(self):
        """RAG composes a long string; name regex must not break on the appended university line."""
        raw = "who is Ahmet bulut"
        composed = (
            f"{raw}\n"
            "Acıbadem Mehmet Ali Aydınlar University (ACU)"
        )
        self.assertEqual(
            whois_name_from_queries(composed, raw),
            "Ahmet bulut",
        )

    def test_whois_vector_variants(self):
        v = whois_vector_variants("Ahmet Bulut", max_variants=2)
        self.assertEqual(len(v), 2)
        self.assertIn("head of department", v[0].lower())
        self.assertIn("ahmet", v[0].lower())

    def test_snippet_around_phrase_centers_name(self):
        long_pre = "x" * 1200
        name = "Mahsa Ziraksima"
        long_post = " y" * 200
        blob = f"{long_pre} {name} is faculty.{long_post}"
        sn = snippet_around_phrase(blob, name, max_len=400)
        self.assertIn(name, sn)
        self.assertNotEqual(sn, blob[:400])

    def test_whois_name_in_content_tokenwise(self):
        self.assertTrue(
            whois_name_in_content("Committee: Mahsa A. Ziraksima (member)", "mahsa ziraksima")
        )
        self.assertTrue(whois_name_in_content("M. Ziraksima and Mahsa", "mahsa ziraksima"))

    def test_whois_folds_turkish_surname(self):
        self.assertTrue(
            whois_name_in_content("Öğr. Gör. Dr. Mahsa Zıraksıma Akreditasyon", "mahsa ziraksima")
        )

    def test_acibadem_js_seed_urls_match_db_normalization(self):
        from core.acibadem_js_scraper import ACIBADEM_JS_URLS

        for u in ACIBADEM_JS_URLS:
            n = normalize_url(u)
            self.assertTrue(
                n.startswith("https://acibadem.edu.tr/en/"),
                msg=n,
            )
            self.assertIn("academic-staff", n)

    def test_faculty_roster_path_filter(self):
        from core.rag_keywords import faculty_list_embedding_phrase, faculty_roster_path_filter

        self.assertEqual(
            faculty_roster_path_filter("Who are the computer engineering teachers?"),
            "computer-engineering",
        )
        self.assertIsNotNone(
            faculty_list_embedding_phrase("list of computer engineering faculty members")
        )


if __name__ == "__main__":
    unittest.main()
