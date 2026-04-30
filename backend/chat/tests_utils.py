"""
Unit tests for chat utility functions — message trimming, client ID parsing,
and LLM response sanitization.

Pure Python — no Django or database setup needed.  Run with:

    python -m unittest chat.tests_utils -v
"""

import unittest
import uuid

from chat.llm_service import _sanitize_assistant_reply
from chat.message_utils import (
    parse_client_id,
    trim_last_user_for_llm,
    trim_message_for_llm,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. parse_client_id
# ═══════════════════════════════════════════════════════════════════════


class TestParseClientId(unittest.TestCase):

    def test_valid_uuid(self):
        uid = uuid.uuid4()
        result = parse_client_id(str(uid))
        self.assertEqual(result, uid)

    def test_valid_uuid_with_hyphens(self):
        raw = "12345678-1234-5678-1234-567812345678"
        result = parse_client_id(raw)
        self.assertIsNotNone(result)

    def test_none_input(self):
        self.assertIsNone(parse_client_id(None))

    def test_empty_string(self):
        self.assertIsNone(parse_client_id(""))

    def test_invalid_uuid(self):
        self.assertIsNone(parse_client_id("not-a-uuid"))

    def test_garbage(self):
        self.assertIsNone(parse_client_id("xyz123"))

    def test_integer(self):
        self.assertIsNone(parse_client_id("42"))


# ═══════════════════════════════════════════════════════════════════════
# 2. trim_message_for_llm
# ═══════════════════════════════════════════════════════════════════════


class TestTrimMessageForLlm(unittest.TestCase):

    def test_short_message_unchanged(self):
        self.assertEqual(trim_message_for_llm("hello", 100), "hello")

    def test_exact_limit(self):
        msg = "a" * 100
        self.assertEqual(trim_message_for_llm(msg, 100), msg)

    def test_truncated_with_ellipsis(self):
        msg = "a" * 200
        result = trim_message_for_llm(msg, 100)
        self.assertEqual(len(result), 100)
        self.assertTrue(result.endswith("…"))

    def test_strips_whitespace(self):
        self.assertEqual(trim_message_for_llm("  hello  ", 100), "hello")

    def test_empty_string(self):
        self.assertEqual(trim_message_for_llm("", 100), "")

    def test_none_input(self):
        self.assertEqual(trim_message_for_llm(None, 100), "")


# ═══════════════════════════════════════════════════════════════════════
# 3. trim_last_user_for_llm
# ═══════════════════════════════════════════════════════════════════════


class TestTrimLastUserForLlm(unittest.TestCase):

    def test_short_message_no_context(self):
        result = trim_last_user_for_llm("hello", 500)
        self.assertEqual(result, "hello")

    def test_with_context_marker_short(self):
        content = "===CONTEXT=== some context here"
        result = trim_last_user_for_llm(content, 5000)
        self.assertEqual(result, content)

    def test_with_context_marker_long(self):
        content = "===CONTEXT===" + "x" * 6000
        result = trim_last_user_for_llm(content, 500)
        self.assertTrue(result.endswith("…"))
        self.assertLessEqual(len(result), 500 + 500)

    def test_without_context_uses_trim(self):
        content = "x" * 2000
        result = trim_last_user_for_llm(content, 500)
        self.assertTrue(result.endswith("…"))


# ═══════════════════════════════════════════════════════════════════════
# 4. _sanitize_assistant_reply
# ═══════════════════════════════════════════════════════════════════════


class TestSanitizeAssistantReply(unittest.TestCase):

    def test_clean_text_unchanged(self):
        self.assertEqual(
            _sanitize_assistant_reply("Hello, how can I help?"),
            "Hello, how can I help?",
        )

    def test_removes_context_marker(self):
        text = "Here is info ===CONTEXT=== about programs"
        result = _sanitize_assistant_reply(text)
        self.assertNotIn("===CONTEXT===", result)

    def test_removes_question_marker(self):
        text = "The answer ===QUESTION=== is 42"
        result = _sanitize_assistant_reply(text)
        self.assertNotIn("===QUESTION===", result)

    def test_removes_end_question_marker(self):
        text = "Info here ===END_QUESTION=== done"
        result = _sanitize_assistant_reply(text)
        self.assertNotIn("===END_QUESTION===", result)

    def test_replaces_according_to_context(self):
        text = "According to, ===CONTEXT=== the tuition is $5000"
        result = _sanitize_assistant_reply(text)
        self.assertNotIn("===CONTEXT===", result)
        self.assertIn("Based on the university website", result)

    def test_replaces_based_on_context(self):
        text = "Based on, ===CONTEXT=== the campus is in Istanbul"
        result = _sanitize_assistant_reply(text)
        self.assertNotIn("===CONTEXT===", result)
        self.assertIn("Based on the university website", result)

    def test_collapses_multiple_spaces(self):
        text = "Hello    world     today"
        result = _sanitize_assistant_reply(text)
        self.assertNotIn("    ", result)
        self.assertEqual(result, "Hello world today")

    def test_collapses_multiple_newlines(self):
        text = "Hello\n\n\n\n\nworld"
        result = _sanitize_assistant_reply(text)
        self.assertNotIn("\n\n\n", result)

    def test_strips_whitespace(self):
        text = "   some reply   "
        result = _sanitize_assistant_reply(text)
        self.assertEqual(result, "some reply")

    def test_empty_string(self):
        self.assertEqual(_sanitize_assistant_reply(""), "")

    def test_none_input(self):
        self.assertEqual(_sanitize_assistant_reply(None), "")

    def test_according_to_comma_start(self):
        text = "According to, the program offers..."
        result = _sanitize_assistant_reply(text)
        self.assertIn("Based on the university website", result)


if __name__ == "__main__":
    unittest.main()
