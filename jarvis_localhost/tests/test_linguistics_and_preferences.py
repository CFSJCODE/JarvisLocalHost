"""Unit and integration tests for linguistic extraction, multi-vectors, and preference memory."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from jarvis_localhost.linguistics import (
    RHETORICAL_VECTOR_DIM,
    STYLE_VECTOR_DIM,
    LinguisticExtractor,
    RhetoricalRole,
    analyze_discourse,
    analyze_lexical,
    analyze_syntax,
    classify_rhetoric,
)
from jarvis_localhost.planning import QueryIntent, ResponsePlanner
from jarvis_localhost.retrieval.vector_store import VectorStore
from jarvis_localhost.storage.database import JarvisDB


class LinguisticsAndPreferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_jarvis.db"
        self.db = JarvisDB(self.db_path)
        self.extractor = LinguisticExtractor()
        self.planner = ResponsePlanner()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_lexical_analysis_computes_ttr_and_yule_k(self) -> None:
        text = (
            "A cinemática direta dos manipuladores robóticos calcula a posição "
            "do efetuador final a partir das variáveis angulares das juntas."
        )
        lex = analyze_lexical(text)
        self.assertGreater(lex.token_count, 10)
        self.assertGreater(lex.ttr, 0.7)
        self.assertGreaterEqual(lex.yule_k, 0.0)
        self.assertGreater(lex.mean_word_length, 3.0)

    def test_syntactic_analysis_captures_punctuation_and_math(self) -> None:
        text = (
            "Considere a seguinte equação matricial: T = A_1 * A_2 * A_3; "
            "onde cada matriz A_i representa a transformação homogênea."
        )
        syn = analyze_syntax(text)
        self.assertGreaterEqual(syn.sentence_count, 1)
        self.assertGreater(syn.colon_density, 0.0)
        self.assertGreater(syn.semicolon_density, 0.0)
        self.assertGreater(syn.math_density, 0.0)

    def test_discourse_analysis_detects_portuguese_connectives(self) -> None:
        text = (
            "Em primeiro lugar, calculamos a matriz de rotação. Além disso, "
            "determinamos o vetor de translação. Entretanto, a cinemática inversa "
            "exige a inversão algébrica. Portanto, a solução pode não ser única."
        )
        disc = analyze_discourse(text)
        self.assertGreaterEqual(disc.total_connective_count, 4)
        self.assertGreater(disc.sequential_count, 0)
        self.assertGreater(disc.additive_count, 0)
        self.assertGreater(disc.adversative_count, 0)
        self.assertGreater(disc.consecutive_count, 0)

    def test_rhetorical_classification(self) -> None:
        def_text = "Cinemática é a ciência que trata do movimento sem considerar as forças."
        comp_text = "Em contraste com a cinemática direta, a cinemática inversa calcula os ângulos."
        proc_text = "Para calcular a posição, em primeiro lugar determinamos as matrizes e em seguida multiplicamos."
        math_text = "Dada a equação matricial de transformação homogênea T com derivada e integral."

        rhet_def = classify_rhetoric(def_text)
        self.assertEqual(rhet_def.primary_role, RhetoricalRole.DEFINITION)

        rhet_comp = classify_rhetoric(comp_text)
        self.assertEqual(rhet_comp.primary_role, RhetoricalRole.COMPARISON)

        rhet_proc = classify_rhetoric(proc_text)
        self.assertEqual(rhet_proc.primary_role, RhetoricalRole.PROCEDURE)

        rhet_math = classify_rhetoric(math_text)
        self.assertEqual(rhet_math.primary_role, RhetoricalRole.MATHEMATICAL_FORMULATION)

    def test_unified_extractor_produces_normalized_orthogonal_vectors(self) -> None:
        sample = (
            "A cinemática inversa consiste em encontrar os ângulos das juntas "
            "que posicionam o efetuador na pose desejada. Portanto, trata-se de um problema não linear."
        )
        profile = self.extractor.extract(sample)
        self.assertEqual(len(profile.style_vector), STYLE_VECTOR_DIM)
        self.assertEqual(len(profile.rhetorical_vector), RHETORICAL_VECTOR_DIM)

        s_norm = np.linalg.norm(profile.style_vector)
        r_norm = np.linalg.norm(profile.rhetorical_vector)
        self.assertAlmostEqual(s_norm, 1.0, places=4)
        self.assertAlmostEqual(r_norm, 1.0, places=4)

    def test_multivector_store_search(self) -> None:
        store = VectorStore()
        c_vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        s_vecs = np.array([
            [0.8, 0.2, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.1, 0.9],
        ], dtype=np.float32)
        r_vecs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ], dtype=np.float32)
        metadata = [
            {"chunk_id": "c1", "rhetorical_role": "definition"},
            {"chunk_id": "c2", "rhetorical_role": "procedure"},
            {"chunk_id": "c3", "rhetorical_role": "definition"},
        ]
        store.add(c_vecs, metadata, style_vectors=s_vecs, rhetorical_vectors=r_vecs)
        self.assertEqual(len(store), 3)

        # Multi-vector query favoring definition
        results = store.search_multi(
            query_vector=np.array([1.0, 0.0, 0.0]),
            target_style_vector=np.array([0.8, 0.2, 0.0]),
            target_rhetorical_vector=np.array([1.0, 0.0]),
            top_k=2,
            rhetorical_role_filter="definition",
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertEqual(results[1]["chunk_id"], "c3")

    def test_response_planner_intents_and_outlines(self) -> None:
        plan_comp = self.planner.plan("Qual a diferença entre cinemática direta e cinemática inversa?")
        self.assertEqual(plan_comp.intent, QueryIntent.COMPARATIVE_ANALYSIS)
        self.assertGreaterEqual(len(plan_comp.sections), 2)

        plan_proc = self.planner.plan("Como calcular a matriz Jacobiana passo a passo?")
        self.assertEqual(plan_proc.intent, QueryIntent.PROCEDURAL_METHOD)

        plan_math = self.planner.plan("Qual a equação e formulação matemática da dinâmica de Newton-Euler?")
        self.assertEqual(plan_math.intent, QueryIntent.MATHEMATICAL_FORMULATION)

        plan_def = self.planner.plan("O que é robótica industrial?")
        self.assertEqual(plan_def.intent, QueryIntent.CONCEPTUAL_DEFINITION)

    def test_database_linguistics_and_preference_pairs(self) -> None:
        # Save linguistic profile
        profile = self.extractor.extract("Definição de manipulador.")
        self.db.save_linguistic_profiles_batch([("chk_test1", "doc_test1", profile)])
        retrieved = self.db.get_linguistic_profile("chk_test1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["document_id"], "doc_test1")

        summary = self.db.get_linguistics_summary()
        self.assertEqual(summary["total_analyzed_chunks"], 1)

        # Save interaction feedback
        self.db.save_interaction_feedback(
            interaction_id="int_001",
            session_id="sess_001",
            question="O que é robô?",
            answer="Um manipulador reprogramável.",
            accepted=True,
            explicit_rating=5,
        )

        # Save preference pair
        pair_id = self.db.save_preference_pair(
            prompt="O que é cinemática?",
            chosen_answer="• **Cinemática:** Estudo do movimento...",
            rejected_answer="É algo de física.",
            reward_delta=1.5,
        )
        self.assertTrue(pair_id.startswith("pref_"))
        pairs = self.db.get_preference_pairs()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["prompt"], "O que é cinemática?")
        self.assertEqual(pairs[0]["reward_delta"], 1.5)


if __name__ == "__main__":
    unittest.main()
