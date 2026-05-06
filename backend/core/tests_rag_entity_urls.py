"""Tests for OBS programme URL entity matching (Django + PostgreSQL)."""

import uuid

from django.test import SimpleTestCase, TestCase

from core.models import DocumentChunk, Page
from core.rag_retrieval import (
    _academic_obs_fallback_chunks,
    _detect_query_intents,
    extract_ects_from_merged_course_text,
    infer_ects_course_anchor,
    infer_curriculum_semester_number,
    _obs_entity_source_url_q,
    _obs_url_matches_target_entity,
)


class TestInferCurriculumSemesterNumber(SimpleTestCase):
    def test_numeric_and_ordinal_semester(self):
        self.assertEqual(
            infer_curriculum_semester_number(
                "computer engineering 4. semester lessons",
            ),
            4,
        )
        self.assertEqual(infer_curriculum_semester_number("4th semester courses"), 4)
        self.assertEqual(infer_curriculum_semester_number("semester 5 plan"), 5)
        self.assertEqual(infer_curriculum_semester_number("first semester"), 1)
        self.assertEqual(
            infer_curriculum_semester_number("computer engineering 4..Semester Course Plan"),
            4,
        )
        self.assertEqual(
            infer_curriculum_semester_number("computer engineering semester-4 lessons"),
            4,
        )


class TestEctsCourseParsers(SimpleTestCase):
    def test_anchor_cleaning_keeps_only_course_name(self):
        q = "computer engineering 4. Semester Course Plan and ects of web programming"
        self.assertEqual(infer_ects_course_anchor(q), "web programming")
        self.assertEqual(infer_ects_course_anchor("what is ects of Web Programming?"), "Web Programming")
        self.assertEqual(
            infer_ects_course_anchor(
                "computer engineering 4. Semester Course Plan and ects of web programming\n"
                "Acıbadem Mehmet Ali Aydınlar University (ACU)",
            ),
            "web programming",
        )

    def test_row_parser_uses_compulsory_column_value(self):
        merged = (
            "4.Semester Course Plan\n"
            "CSE 220 Web Programming 3+0+0 Compulsory 6 Face To Face\n"
            "Total ECTS 30\n"
        )
        row, val = extract_ects_from_merged_course_text(merged, "Web Programming")
        self.assertIn("Web Programming", row)
        self.assertEqual(val, "6")

    def test_row_parser_does_not_take_total_ects(self):
        merged = (
            "4.Semester Course Plan\n"
            "CSE 220 Web Programming 3+0+0 Compulsory Face To Face\n"
            "Total ECTS 30\n"
        )
        row, val = extract_ects_from_merged_course_text(merged, "Web Programming")
        self.assertEqual(row, "")
        self.assertIsNone(val)

    def test_row_parser_handles_flattened_text_blob(self):
        merged = (
            "4.Semester Course Plan CSE 220 Web Programming 3+0+0 Compulsory 6 Face To Face "
            "Total ECTS 36"
        )
        row, val = extract_ects_from_merged_course_text(merged, "Web Programming")
        self.assertIn("Web Programming", row)
        self.assertEqual(val, "6")

    def test_row_parser_handles_of_for_wording_difference(self):
        merged = (
            "3.Semester Course Plan\n"
            "MEG 207 Economics for Engineering 3+0+0 Compulsory 5 Face To Face\n"
            "Total ECTS 29\n"
        )
        row, val = extract_ects_from_merged_course_text(merged, "economics of engineering")
        self.assertIn("Economics for Engineering", row)
        self.assertEqual(val, "5")


class TestObsEntityUrlMatch(TestCase):
    def test_curunit_14_matches_computer_engineering_intent(self):
        intents = _detect_query_intents(
            "Computer Engineering undergraduate curriculum ECTS courses ACU",
            "What are the ECTS credits?",
        )
        self.assertEqual(intents.target_entity, "computer-engineering")
        url = (
            "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?"
            "lang=en&curOp=showPac&curUnit=14&curSunit=6246"
        )
        self.assertTrue(_obs_url_matches_target_entity(url, intents))

    def test_wrong_curunit_does_not_match(self):
        intents = _detect_query_intents(
            "Computer Engineering curriculum ECTS", None
        )
        url = (
            "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?"
            "lang=en&curOp=showPac&curUnit=09&curSunit=5848"
        )
        self.assertFalse(_obs_url_matches_target_entity(url, intents))

    def test_obs_entity_source_url_q_builds_or_filter(self):
        from core.rag_retrieval import _OBS_ENTITY_CODE_HINTS

        hints = _OBS_ENTITY_CODE_HINTS["computer-engineering"]
        q = _obs_entity_source_url_q(hints)
        self.assertIsNotNone(q)
        n = DocumentChunk.objects.filter(q).count()
        if n == 0:
            self.skipTest("No OBS chunks in DB; URL Q shape still covered by unit tests above")


class TestAcademicObsFallbackChunks(TestCase):
    databases = {"default"}

    @classmethod
    def setUpTestData(cls):
        suffix = uuid.uuid4().hex[:8]
        cls.ce_url = (
            "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?"
            f"lang=en&curOp=showPac&curUnit=14&curSunit=6246&_test={suffix}a"
        )
        cls.other_url = (
            "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?"
            f"lang=en&curOp=showPac&curUnit=99&curSunit=99999&_test={suffix}b"
        )
        # Minimal 384-dim zero vector for pgvector NOT NULL (matches project embedding dim)
        dim = 384
        z = [0.0] * dim
        p1 = Page.objects.create(
            url=cls.ce_url,
            title="CE programme",
            content="x",
            source="obs.acibadem.edu.tr",
        )
        p2 = Page.objects.create(
            url=cls.other_url,
            title="Other programme",
            content="y",
            source="obs.acibadem.edu.tr",
        )
        DocumentChunk.objects.create(
            page=p1,
            chunk_index=0,
            content="Computer Engineering semester modules ECTS",
            embedding=z,
            source_url=cls.ce_url,
            page_title="CE programme",
        )
        DocumentChunk.objects.create(
            page=p2,
            chunk_index=0,
            content="Unrelated programme body text without CE name",
            embedding=z,
            source_url=cls.other_url,
            page_title="Other programme",
        )

    def test_fallback_prefers_entity_scoped_rows(self):
        intents = _detect_query_intents(
            "Computer Engineering courses and ECTS Bologna OBS", None
        )
        self.assertTrue(intents.academic_obs)
        out = _academic_obs_fallback_chunks(intents, limit=2)
        urls = [c.source_url for c in out]
        self.assertTrue(any(self.ce_url in (u or "") for u in urls))
        self.assertLessEqual(len(out), 2)
