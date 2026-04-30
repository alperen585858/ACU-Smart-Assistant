"""
Unit tests for core.rag_keywords — intent detection, entity aliasing, keyword
extraction, and embedding phrase generators.

All functions in this module are pure Python (regex + string ops), so no Django
or database setup is needed.  Run with:

    python -m unittest core.tests_rag_keywords -v
"""

import unittest

from core.rag_keywords import (
    RAG_ACADEMIC_OBS_INTENT_RE,
    RAG_BROAD_FEE_LIST_INTENT_RE,
    RAG_DEPT_OR_FACULTY_INTENT_RE,
    RAG_FACULTY_ROSTER_INTENT_RE,
    RAG_FEE_TUITION_INTENT_RE,
    RAG_GRADUATE_ADMISSIONS_INTENT_RE,
    RAG_LEADERSHIP_INTENT_RE,
    RAG_LOCATION_CONTACT_INTENT_RE,
    RAG_STEM_OR_ENGINEERING_INTENT_RE,
    department_snippet_anchor_phrases,
    extract_target_entity_key,
    faculty_list_embedding_phrase,
    faculty_roster_path_filter,
    fee_snippet_anchor_phrases,
    fee_tuition_intent,
    graduate_or_postgrad_admissions_intent,
    international_admissions_default_undergraduate_only,
    international_admissions_embedding_phrase,
    international_application_requirements_page_intent,
    international_student_apply_intent,
    is_university_wide_fee_rag_query,
    leadership_embedding_phrase,
    rag_keywords_from_query,
    stem_engineering_boost_terms,
    structured_list_boost_terms,
    target_entity_aliases,
    target_entity_competitor_aliases,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. Regex pattern tests
# ═══════════════════════════════════════════════════════════════════════


class TestDeptFacultyIntentRegex(unittest.TestCase):
    """RAG_DEPT_OR_FACULTY_INTENT_RE matches department/faculty queries."""

    def _match(self, text: str) -> bool:
        return bool(RAG_DEPT_OR_FACULTY_INTENT_RE.search(text))

    def test_english_department(self):
        self.assertTrue(self._match("What departments are there?"))

    def test_english_faculty(self):
        self.assertTrue(self._match("Tell me about the faculties"))

    def test_turkish_fakulte(self):
        self.assertTrue(self._match("Fakülte listesi nedir?"))

    def test_turkish_bolum(self):
        self.assertTrue(self._match("Bölümler hakkında bilgi"))

    def test_program_list(self):
        self.assertTrue(self._match("Can I see the program list?"))

    def test_engineering_turkish(self):
        self.assertTrue(self._match("Mühendislik bölümü hakkında"))

    def test_no_match(self):
        self.assertFalse(self._match("What is the weather today?"))


class TestStemEngineeringIntentRegex(unittest.TestCase):
    """RAG_STEM_OR_ENGINEERING_INTENT_RE matches STEM queries."""

    def _match(self, text: str) -> bool:
        return bool(RAG_STEM_OR_ENGINEERING_INTENT_RE.search(text))

    def test_computer(self):
        self.assertTrue(self._match("computer engineering program"))

    def test_software(self):
        self.assertTrue(self._match("software engineering"))

    def test_electrical(self):
        self.assertTrue(self._match("electrical engineering department"))

    def test_turkish_bilgisayar(self):
        self.assertTrue(self._match("bilgisayar mühendisliği"))

    def test_turkish_yazilim(self):
        self.assertTrue(self._match("yazılım mühendisliği programı"))

    def test_no_match_medicine(self):
        self.assertFalse(self._match("faculty of medicine"))

    def test_no_match_generic(self):
        self.assertFalse(self._match("campus location"))


class TestLocationContactIntentRegex(unittest.TestCase):
    """RAG_LOCATION_CONTACT_INTENT_RE matches location/contact queries."""

    def _match(self, text: str) -> bool:
        return bool(RAG_LOCATION_CONTACT_INTENT_RE.search(text))

    def test_where_is(self):
        self.assertTrue(self._match("Where is the campus?"))

    def test_address(self):
        self.assertTrue(self._match("What is the address?"))

    def test_contact(self):
        self.assertTrue(self._match("Contact information please"))

    def test_turkish_nerede(self):
        self.assertTrue(self._match("Üniversite nerede?"))

    def test_turkish_iletisim(self):
        self.assertTrue(self._match("İletişim bilgileri"))

    def test_transportation(self):
        self.assertTrue(self._match("How to get to the campus? transportation"))

    def test_no_match(self):
        self.assertFalse(self._match("What programs do you offer?"))


class TestAcademicObsIntentRegex(unittest.TestCase):
    """RAG_ACADEMIC_OBS_INTENT_RE matches OBS/Bologna queries."""

    def _match(self, text: str) -> bool:
        return bool(RAG_ACADEMIC_OBS_INTENT_RE.search(text))

    def test_obs(self):
        self.assertTrue(self._match("How do I access OBS?"))

    def test_bologna(self):
        self.assertTrue(self._match("Bologna course catalog"))

    def test_curriculum(self):
        self.assertTrue(self._match("Show me the curriculum"))

    def test_ects(self):
        self.assertTrue(self._match("How many ECTS credits?"))

    def test_turkish_ders(self):
        self.assertTrue(self._match("Dersler listesi"))

    def test_no_match(self):
        self.assertFalse(self._match("How much is the tuition?"))


class TestFacultyRosterIntentRegex(unittest.TestCase):
    """RAG_FACULTY_ROSTER_INTENT_RE matches teacher/staff list queries."""

    def _match(self, text: str) -> bool:
        return bool(RAG_FACULTY_ROSTER_INTENT_RE.search(text))

    def test_teachers(self):
        self.assertTrue(self._match("Who are the teachers?"))

    def test_professors(self):
        self.assertTrue(self._match("List of professors"))

    def test_academic_staff(self):
        self.assertTrue(self._match("academic staff of engineering"))

    def test_turkish_ogretim(self):
        self.assertTrue(self._match("öğretim üyeleri kimler?"))

    def test_turkish_kadro(self):
        self.assertTrue(self._match("Akademik kadro listesi"))

    def test_no_match(self):
        self.assertFalse(self._match("How much is the tuition?"))


class TestLeadershipIntentRegex(unittest.TestCase):
    """RAG_LEADERSHIP_INTENT_RE matches dean/rector queries."""

    def _match(self, text: str) -> bool:
        return bool(RAG_LEADERSHIP_INTENT_RE.search(text))

    def test_dean(self):
        self.assertTrue(self._match("Who is the dean?"))

    def test_rector(self):
        self.assertTrue(self._match("Who is the rector of the university?"))

    def test_turkish_dekan(self):
        self.assertTrue(self._match("Dekan kim?"))

    def test_turkish_rektor(self):
        self.assertTrue(self._match("Rektör hakkında bilgi"))

    def test_no_match(self):
        self.assertFalse(self._match("List of departments"))


class TestFeeTuitionIntentRegex(unittest.TestCase):
    """RAG_FEE_TUITION_INTENT_RE matches fee/tuition queries."""

    def _match(self, text: str) -> bool:
        return bool(RAG_FEE_TUITION_INTENT_RE.search(text))

    def test_tuition(self):
        self.assertTrue(self._match("What is the tuition?"))

    def test_fees(self):
        self.assertTrue(self._match("How much are the fees?"))

    def test_how_much(self):
        self.assertTrue(self._match("How much does it cost?"))

    def test_scholarship(self):
        self.assertTrue(self._match("Are there scholarship opportunities?"))

    def test_turkish_ucret(self):
        self.assertTrue(self._match("Öğrenim ücretleri nedir?"))

    def test_turkish_burs(self):
        self.assertTrue(self._match("Burs imkanları"))

    def test_no_match(self):
        self.assertFalse(self._match("Where is the campus?"))


class TestGraduateAdmissionsIntentRegex(unittest.TestCase):
    """RAG_GRADUATE_ADMISSIONS_INTENT_RE matches graduate-level queries."""

    def _match(self, text: str) -> bool:
        return bool(RAG_GRADUATE_ADMISSIONS_INTENT_RE.search(text))

    def test_masters(self):
        self.assertTrue(self._match("master's program admission"))

    def test_phd(self):
        self.assertTrue(self._match("PhD application requirements"))

    def test_graduate(self):
        self.assertTrue(self._match("graduate school admission"))

    def test_turkish_yuksek_lisans(self):
        self.assertTrue(self._match("yüksek lisans başvurusu"))

    def test_turkish_doktora(self):
        self.assertTrue(self._match("doktora programı"))

    def test_no_match_undergrad(self):
        self.assertFalse(self._match("undergraduate programs"))


class TestBroadFeeListIntentRegex(unittest.TestCase):
    """RAG_BROAD_FEE_LIST_INTENT_RE matches university-wide fee queries."""

    def _match(self, text: str) -> bool:
        return bool(RAG_BROAD_FEE_LIST_INTENT_RE.search(text))

    def test_all_fees(self):
        self.assertTrue(self._match("What are all the fees?"))

    def test_all_programs(self):
        self.assertTrue(self._match("Fees for all programs"))

    def test_turkish_tum_ucret(self):
        self.assertTrue(self._match("Tüm ücretler ne?"))

    def test_fee_list(self):
        self.assertTrue(self._match("Show me the fee list"))

    def test_no_match_single_dept(self):
        self.assertFalse(self._match("Computer engineering tuition"))


# ═══════════════════════════════════════════════════════════════════════
# 2. Entity aliasing functions
# ═══════════════════════════════════════════════════════════════════════


class TestExtractTargetEntityKey(unittest.TestCase):

    def test_computer_engineering(self):
        self.assertEqual(
            extract_target_entity_key("computer engineering department"),
            "computer-engineering",
        )

    def test_computer_programming(self):
        self.assertEqual(
            extract_target_entity_key("computer programming"),
            "computer-programming",
        )

    def test_health_sciences(self):
        self.assertEqual(
            extract_target_entity_key("faculty of health sciences tuition"),
            "faculty-of-health-sciences",
        )

    def test_medicine(self):
        self.assertEqual(
            extract_target_entity_key("faculty of medicine"),
            "faculty-of-medicine",
        )

    def test_turkish_bilgisayar(self):
        self.assertEqual(
            extract_target_entity_key("bilgisayar mühendisliği bölümü"),
            "computer-engineering",
        )

    def test_no_match(self):
        self.assertIsNone(extract_target_entity_key("campus location"))

    def test_empty(self):
        self.assertIsNone(extract_target_entity_key(""))

    def test_none(self):
        self.assertIsNone(extract_target_entity_key(None))

    def test_priority_programming_over_engineering(self):
        # "computer programming" should match before "computer engineering"
        # because it comes first in _ENTITY_ORDER
        self.assertEqual(
            extract_target_entity_key("computer programming"),
            "computer-programming",
        )


class TestTargetEntityAliases(unittest.TestCase):

    def test_known_key(self):
        aliases = target_entity_aliases("computer-engineering")
        self.assertIn("computer engineering", aliases)
        self.assertIn("bilgisayar mühendisliği", aliases)

    def test_medicine_key(self):
        aliases = target_entity_aliases("faculty-of-medicine")
        self.assertIn("faculty of medicine", aliases)
        self.assertIn("tıp fakültesi", aliases)

    def test_none_key(self):
        self.assertEqual(target_entity_aliases(None), ())

    def test_unknown_key(self):
        self.assertEqual(target_entity_aliases("unknown-dept"), ())


class TestTargetEntityCompetitorAliases(unittest.TestCase):

    def test_excludes_own_aliases(self):
        competitors = target_entity_competitor_aliases("computer-engineering")
        self.assertNotIn("computer engineering", competitors)
        # But should include other entities
        self.assertIn("faculty of medicine", competitors)
        self.assertIn("computer programming", competitors)

    def test_none_key(self):
        self.assertEqual(target_entity_competitor_aliases(None), ())


# ═══════════════════════════════════════════════════════════════════════
# 3. Faculty roster path filtering
# ═══════════════════════════════════════════════════════════════════════


class TestFacultyRosterPathFilter(unittest.TestCase):

    def test_computer_engineering(self):
        self.assertEqual(
            faculty_roster_path_filter("computer engineering professors"),
            "computer-engineering",
        )

    def test_computer_programming(self):
        self.assertEqual(
            faculty_roster_path_filter("computer programming teachers"),
            "computer-programming",
        )

    def test_medicine(self):
        self.assertEqual(
            faculty_roster_path_filter("faculty of medicine"),
            "faculty-of-medicine",
        )

    def test_health_sciences(self):
        self.assertEqual(
            faculty_roster_path_filter("health sciences faculty"),
            "faculty-of-health-sciences",
        )

    def test_electrical(self):
        self.assertEqual(
            faculty_roster_path_filter("electrical engineering"),
            "electrical",
        )

    def test_mechanical(self):
        self.assertEqual(
            faculty_roster_path_filter("mechanical engineering"),
            "mechanical",
        )

    def test_software(self):
        self.assertEqual(
            faculty_roster_path_filter("software engineering"),
            "software",
        )

    def test_medicine_fee_maps_to_medicine(self):
        self.assertEqual(
            faculty_roster_path_filter("medicine tuition fee"),
            "faculty-of-medicine",
        )

    def test_biomedical_not_medicine(self):
        # "biomedical" should map to biomedical, not faculty-of-medicine
        self.assertEqual(
            faculty_roster_path_filter("biomedical engineering"),
            "biomedical",
        )

    def test_no_match(self):
        self.assertIsNone(faculty_roster_path_filter("campus location"))

    def test_empty(self):
        self.assertIsNone(faculty_roster_path_filter(""))

    def test_none(self):
        self.assertIsNone(faculty_roster_path_filter(None))


# ═══════════════════════════════════════════════════════════════════════
# 4. Intent detection functions
# ═══════════════════════════════════════════════════════════════════════


class TestFeeTuitionIntent(unittest.TestCase):

    def test_positive_english(self):
        self.assertTrue(fee_tuition_intent("How much is the tuition?"))

    def test_positive_turkish(self):
        self.assertTrue(fee_tuition_intent("Öğrenim ücreti nedir?"))

    def test_positive_scholarship(self):
        self.assertTrue(fee_tuition_intent("scholarship opportunities"))

    def test_negative(self):
        self.assertFalse(fee_tuition_intent("Where is the campus?"))

    def test_empty(self):
        self.assertFalse(fee_tuition_intent(""))

    def test_none(self):
        self.assertFalse(fee_tuition_intent(None))


class TestGraduateOrPostgradAdmissionsIntent(unittest.TestCase):

    def test_masters(self):
        self.assertTrue(
            graduate_or_postgrad_admissions_intent("master's admission")
        )

    def test_phd(self):
        self.assertTrue(
            graduate_or_postgrad_admissions_intent("PhD program requirements")
        )

    def test_yuksek_lisans(self):
        self.assertTrue(
            graduate_or_postgrad_admissions_intent("yüksek lisans başvuru")
        )

    def test_not_graduate(self):
        self.assertFalse(
            graduate_or_postgrad_admissions_intent("undergraduate programs")
        )

    def test_empty(self):
        self.assertFalse(graduate_or_postgrad_admissions_intent(""))


class TestInternationalStudentApplyIntent(unittest.TestCase):

    def test_international_application(self):
        self.assertTrue(
            international_student_apply_intent(
                "How do international students apply?"
            )
        )

    def test_foreign_student_admission(self):
        self.assertTrue(
            international_student_apply_intent(
                "foreign student admission requirements"
            )
        )

    def test_turkish_yabanci_ogrenci(self):
        self.assertTrue(
            international_student_apply_intent(
                "yabancı öğrenci başvuru koşulları"
            )
        )

    def test_not_international(self):
        self.assertFalse(
            international_student_apply_intent("What is the tuition?")
        )

    def test_fee_only_not_apply(self):
        # Pure fee question without apply/admission angle
        self.assertFalse(
            international_student_apply_intent(
                "international student tuition how much"
            )
        )

    def test_empty(self):
        self.assertFalse(international_student_apply_intent(""))


class TestInternationalApplicationRequirementsPageIntent(unittest.TestCase):

    def test_requirements_query(self):
        self.assertTrue(
            international_application_requirements_page_intent(
                "What are the requirements for international student admission?"
            )
        )

    def test_exam_scores(self):
        self.assertTrue(
            international_application_requirements_page_intent(
                "What SAT score do international students need to apply?"
            )
        )

    def test_no_requirements(self):
        self.assertFalse(
            international_application_requirements_page_intent(
                "How much is tuition?"
            )
        )


class TestInternationalAdmissionsDefaultUndergraduateOnly(unittest.TestCase):

    def test_ug_international(self):
        self.assertTrue(
            international_admissions_default_undergraduate_only(
                "What are the requirements for international student admission?"
            )
        )

    def test_graduate_not_ug(self):
        self.assertFalse(
            international_admissions_default_undergraduate_only(
                "international student master's program admission"
            )
        )

    def test_empty(self):
        self.assertFalse(
            international_admissions_default_undergraduate_only("")
        )


class TestIsUniversityWideFeeRagQuery(unittest.TestCase):

    def test_all_fees(self):
        self.assertTrue(is_university_wide_fee_rag_query("all fees"))

    def test_fee_list(self):
        self.assertTrue(is_university_wide_fee_rag_query("fee list"))

    def test_general_tuition(self):
        # No specific department → university-wide
        self.assertTrue(is_university_wide_fee_rag_query("How much is tuition?"))

    def test_specific_dept_not_wide(self):
        # Specific department → not university-wide
        self.assertFalse(
            is_university_wide_fee_rag_query("computer engineering tuition fee")
        )

    def test_no_fee_intent(self):
        self.assertFalse(
            is_university_wide_fee_rag_query("Where is the campus?")
        )


# ═══════════════════════════════════════════════════════════════════════
# 5. Embedding phrase generators
# ═══════════════════════════════════════════════════════════════════════


class TestLeadershipEmbeddingPhrase(unittest.TestCase):

    def test_dean_query(self):
        result = leadership_embedding_phrase("Who is the dean?")
        self.assertIsNotNone(result)
        self.assertIn("dean", result.lower())
        self.assertIn("rector", result.lower())

    def test_rector_query(self):
        result = leadership_embedding_phrase("Who is the rector?")
        self.assertIsNotNone(result)

    def test_no_leadership(self):
        self.assertIsNone(leadership_embedding_phrase("computer engineering"))

    def test_empty(self):
        self.assertIsNone(leadership_embedding_phrase(""))

    def test_none(self):
        self.assertIsNone(leadership_embedding_phrase(None))


class TestFacultyListEmbeddingPhrase(unittest.TestCase):

    def test_engineering_teachers(self):
        result = faculty_list_embedding_phrase(
            "computer engineering teachers"
        )
        self.assertIsNotNone(result)
        self.assertIn("academic staff", result.lower())

    def test_no_roster_intent(self):
        self.assertIsNone(
            faculty_list_embedding_phrase("What is tuition?")
        )

    def test_roster_but_no_dept(self):
        # Faculty roster intent but no specific department or STEM
        self.assertIsNone(
            faculty_list_embedding_phrase("teachers list")
        )


class TestInternationalAdmissionsEmbeddingPhrase(unittest.TestCase):

    def test_apply_intent(self):
        result = international_admissions_embedding_phrase(
            "How do international students apply for admission?"
        )
        self.assertIsNotNone(result)
        self.assertIn("international", result.lower())

    def test_requirements_intent(self):
        result = international_admissions_embedding_phrase(
            "What are the requirements for international student admission?"
        )
        self.assertIsNotNone(result)
        self.assertIn("requirements", result.lower())

    def test_no_intent(self):
        self.assertIsNone(
            international_admissions_embedding_phrase("campus location")
        )


# ═══════════════════════════════════════════════════════════════════════
# 6. Keyword extraction
# ═══════════════════════════════════════════════════════════════════════


class TestRagKeywordsFromQuery(unittest.TestCase):

    def test_extracts_keywords(self):
        result = rag_keywords_from_query("computer engineering tuition fees")
        self.assertIn("computer", result)
        self.assertIn("engineering", result)
        self.assertIn("tuition", result)
        self.assertIn("fees", result)

    def test_filters_stopwords(self):
        result = rag_keywords_from_query("what are the programs?")
        self.assertNotIn("what", result)
        self.assertNotIn("the", result)

    def test_max_terms(self):
        result = rag_keywords_from_query(
            "some long query with many different unique important words here",
            max_terms=3,
        )
        self.assertLessEqual(len(result), 3)

    def test_empty(self):
        self.assertEqual(rag_keywords_from_query(""), [])

    def test_none(self):
        self.assertEqual(rag_keywords_from_query(None), [])

    def test_short_words_excluded(self):
        # Words < 4 chars should be excluded
        result = rag_keywords_from_query("I am at ACU now")
        self.assertEqual(result, [])

    def test_deduplication(self):
        result = rag_keywords_from_query("tuition tuition tuition")
        self.assertEqual(result.count("tuition"), 1)


# ═══════════════════════════════════════════════════════════════════════
# 7. STEM boost terms
# ═══════════════════════════════════════════════════════════════════════


class TestStemEngineeringBoostTerms(unittest.TestCase):

    def test_computer_engineering(self):
        terms = stem_engineering_boost_terms("computer engineering department")
        self.assertTrue(any("computer" in t.lower() for t in terms))

    def test_computer_programming(self):
        terms = stem_engineering_boost_terms("computer programming")
        self.assertTrue(
            any("programming" in t.lower() for t in terms)
        )

    def test_electrical(self):
        terms = stem_engineering_boost_terms("electrical engineering")
        self.assertTrue(any("electri" in t.lower() for t in terms))

    def test_mechanical(self):
        terms = stem_engineering_boost_terms("mechanical engineering")
        self.assertTrue(any("mechani" in t.lower() or "makine" in t.lower() for t in terms))

    def test_generic_engineering(self):
        terms = stem_engineering_boost_terms("engineering faculty")
        self.assertTrue(len(terms) > 0)
        self.assertTrue(any("engineering" in t.lower() for t in terms))

    def test_no_stem(self):
        self.assertEqual(stem_engineering_boost_terms("campus location"), [])

    def test_empty(self):
        self.assertEqual(stem_engineering_boost_terms(""), [])

    def test_deduplication(self):
        terms = stem_engineering_boost_terms("computer engineering")
        lowered = [t.lower() for t in terms]
        self.assertEqual(len(lowered), len(set(lowered)))


# ═══════════════════════════════════════════════════════════════════════
# 8. Structured list boost terms
# ═══════════════════════════════════════════════════════════════════════


class TestStructuredListBoostTerms(unittest.TestCase):

    def test_faculty_query(self):
        terms = structured_list_boost_terms("List all faculties")
        self.assertTrue(len(terms) > 0)
        self.assertIn("Faculty", terms)

    def test_department_query(self):
        terms = structured_list_boost_terms("What departments are there?")
        self.assertIn("Department", terms)

    def test_no_dept_intent(self):
        self.assertEqual(
            structured_list_boost_terms("What is the tuition?"), []
        )

    def test_empty(self):
        self.assertEqual(structured_list_boost_terms(""), [])


# ═══════════════════════════════════════════════════════════════════════
# 9. Snippet anchor phrases
# ═══════════════════════════════════════════════════════════════════════


class TestDepartmentSnippetAnchorPhrases(unittest.TestCase):

    def test_computer_engineering(self):
        phrases = department_snippet_anchor_phrases(
            "computer engineering tuition"
        )
        self.assertTrue(len(phrases) > 0)
        # Should include multi-word names and URL-label-style
        lowered = [p.lower() for p in phrases]
        self.assertTrue(
            any("computer" in p for p in lowered)
        )

    def test_no_dept(self):
        self.assertEqual(
            department_snippet_anchor_phrases("campus location"), []
        )


class TestFeeSnippetAnchorPhrases(unittest.TestCase):

    def test_scholarship_query(self):
        phrases = fee_snippet_anchor_phrases("scholarship opportunities fee")
        self.assertTrue(len(phrases) > 0)
        lowered = [p.lower() for p in phrases]
        self.assertTrue(any("scholarship" in p for p in lowered))

    def test_payment_query(self):
        phrases = fee_snippet_anchor_phrases("payment plan tuition")
        self.assertTrue(len(phrases) > 0)
        lowered = [p.lower() for p in phrases]
        self.assertTrue(any("payment" in p for p in lowered))

    def test_no_fee_intent(self):
        self.assertEqual(fee_snippet_anchor_phrases("campus location"), [])


if __name__ == "__main__":
    unittest.main()
