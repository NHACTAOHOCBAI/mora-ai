import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from app.services.benchmark_details import RecordingJudge, details_for


class CalculationTests(unittest.TestCase):
    def outputs(self, schema, data):
        return [{"schema": schema, "data": item} for item in data]

    def test_precision_reconstructs_ragas_average_precision_in_original_order(self):
        from ragas.metrics.collections import ContextPrecision
        for verdicts in ([1, 0, 1], [1, 1, 1], [0, 0, 0]):
            outputs = self.outputs("ContextPrecisionOutput", [{"verdict": v, "reason": f"reason-{i}"} for i, v in enumerate(verdicts)])
            contexts = [{"documentId": 7, "pageNumber": p, "text": str(p)} for p in [3, 1, 2]]
            details = details_for("contextPrecision", outputs, None, contexts, True)
            expected = ContextPrecision(llm=RecordingJudge(MagicMock()))._calculate_average_precision(verdicts)
            self.assertAlmostEqual(details["numerator"] / details["denominator"], expected, places=12)
            self.assertEqual([row["context"]["pageNumber"] for row in details["rows"]], [3, 1, 2])
            self.assertEqual([row["reason"] for row in details["rows"]], ["reason-0", "reason-1", "reason-2"])

    def test_timeout_keeps_only_observed_verdicts_and_marks_incomplete(self):
        details = details_for("contextPrecision", self.outputs("ContextPrecisionOutput", [{"verdict": 1, "reason": "supported"}]),
                              None, [{"text": "A"}, {"text": "B"}], False)
        self.assertFalse(details["complete"])
        self.assertEqual(len(details["rows"]), 1)
        self.assertEqual(details["expectedContexts"], 2)

    def test_faithfulness_preserves_statements_reasons_and_actual_counts(self):
        statements = [{"statement": "A", "reason": "In source", "verdict": 1},
                      {"statement": "B", "reason": "Not in source", "verdict": 0}]
        outputs = self.outputs("StatementGeneratorOutput", [{"statements": ["A", "B"]}]) + self.outputs("NLIStatementOutput", [{"statements": statements}])
        details = details_for("faithfulness", outputs, None, [], True)
        self.assertEqual(details["statements"], statements)
        self.assertEqual(details["supportedCount"] / details["statementCount"], .5)
        partial = details_for("faithfulness", outputs[:1], None, [], False)
        self.assertFalse(partial["complete"])
        self.assertEqual(partial["statementCount"], 0)
        self.assertEqual(partial["generatedStatements"], ["A", "B"])

    def test_relevance_records_cosine_components_and_skips_empty_questions(self):
        outputs = self.outputs("AnswerRelevanceOutput", [
            {"question": "Q1", "noncommittal": 0}, {"question": "", "noncommittal": 1},
            {"question": "Q2", "noncommittal": 1}])
        vectors = SimpleNamespace(original=[3., 4.], generated=[[0., 5.], [-3., -4.]])
        details = details_for("answerRelevance", outputs, vectors, [], True)
        self.assertEqual(details["originalNorm"], 5)
        self.assertEqual(details["rows"][0]["dotProduct"], 20)
        self.assertEqual(details["rows"][0]["cosine"], .8)
        self.assertFalse(details["rows"][1]["used"])
        self.assertEqual(details["rows"][2]["cosine"], -1)
        self.assertAlmostEqual(details["meanCosine"], -.1)
        self.assertEqual(details["usedCount"], 2)
        self.assertEqual(details["substantiveFactor"], 1)
        for item in outputs:
            item["data"]["noncommittal"] = 1
        self.assertEqual(details_for("answerRelevance", outputs, vectors, [], True)["substantiveFactor"], 0)
        self.assertNotIn("generated", details)
        self.assertNotIn("original", details)

    def test_missing_embedding_zero_norm_and_invalid_vectors_do_not_invent_cosines(self):
        outputs = self.outputs("AnswerRelevanceOutput", [{"question": "Q", "noncommittal": 0}])
        for vectors in [SimpleNamespace(original=None, generated=[]),
                        SimpleNamespace(original=[0., 0.], generated=[[0., 0.]]),
                        SimpleNamespace(original=[1., 0.], generated=[[1.]]),
                        SimpleNamespace(original=[float('nan')], generated=[[1.]])]:
            details = details_for("answerRelevance", outputs, vectors, [], False)
            self.assertIsNone(details["rows"][0]["cosine"])
            self.assertIsNone(details["meanCosine"])
            import json
            json.dumps(details, allow_nan=False)


class RecordingTests(unittest.IsolatedAsyncioTestCase):
    async def test_recorder_returns_same_result_and_does_not_add_provider_calls(self):
        class Verdict(BaseModel):
            verdict: int
            reason: str
        result = Verdict(verdict=1, reason="Original judge reason")
        inner = SimpleNamespace(agenerate=AsyncMock(return_value=result))
        recorder = RecordingJudge(inner)
        self.assertIs(await recorder.agenerate("prompt", Verdict), result)
        inner.agenerate.assert_awaited_once_with("prompt", Verdict)
        self.assertEqual(recorder.outputs, [{"schema": "Verdict", "data": result.model_dump()}])
        inner.agenerate.side_effect = TimeoutError()
        with self.assertRaises(TimeoutError):
            await recorder.agenerate("prompt", Verdict)
        self.assertEqual(len(recorder.outputs), 1)
