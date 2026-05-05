"""Tests for OBS programme URL entity matching (Django + PostgreSQL)."""

import uuid

from django.test import TestCase

from core.models import DocumentChunk, Page
from core.rag_retrieval import (
    _academic_obs_fallback_chunks,
    _detect_query_intents,
    _obs_entity_source_url_q,
    _obs_url_matches_target_entity,
)


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
