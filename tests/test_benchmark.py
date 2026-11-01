import asyncio
import unittest
import importlib
import sys
from unittest.mock import patch, MagicMock, AsyncMock
from types import SimpleNamespace
from pydantic import ValidationError
from app.core.execution_trace import execution_trace, event, failure
from app.schemas.chat import ChatRequest
from app.services.chat_service import generate_chat_response
from app.services.benchmark_service import ExecuteSample, ScoreSample, execute, score


class TraceTests(unittest.TestCase):
    def run_pipeline(self, intent="RAG", sufficient=True, faithful=True, unavailable=False):
        trace = {"errors": []}
        token = execution_trace.set(trace)
        try:
            with patch("app.services.chat_service.MultiAgentOrchestrator") as factory:
                agent = factory.return_value
                agent.route_agent.return_value = intent
                agent.retrieval_agent.return_value = [{"text": "first", "pageNumber": 3}, {"text": "second", "pageNumber": 1}]
                agent.synthesis_agent.return_value = {"answer": "answer", "citations": [], "hasSufficientInfo": sufficient}
                agent.evaluator_agent.return_value = (False, None) if unavailable else (faithful, 1 if faithful else 0)
                agent.general_chat_agent.return_value = ("general", "prompt")
                response = generate_chat_response(ChatRequest(question="question", history=[]))
                return trace, response, agent
        finally:
            execution_trace.reset(token)

    def test_rag_preserves_context_order_and_response_contract(self):
        trace, response, _ = self.run_pipeline()
        self.assertEqual([c["text"] for c in trace["pipeline"]["contexts"]], ["first", "second"])
        self.assertFalse(trace["pipeline"]["fallback"])
        self.assertEqual(trace["pipeline"]["synthesisAttempts"], 1)
        self.assertGreaterEqual(trace["pipeline"]["timings"]["totalMs"], 0)
        self.assertEqual(set(response.model_dump()), {"answer", "citations", "condensedQuestion", "promptSent"})

    def test_general_has_no_retrieval(self):
        trace, response, agent = self.run_pipeline(intent="GENERAL")
        agent.retrieval_agent.assert_not_called()
        self.assertEqual(trace["pipeline"]["contexts"], [])
        self.assertEqual(response.answer, "general")

    def test_insufficient_info_falls_back(self):
        trace, response, agent = self.run_pipeline(sufficient=False)
        self.assertTrue(trace["pipeline"]["fallback"])
        agent.evaluator_agent.assert_not_called()
        self.assertIn("general", response.answer)

    def test_failed_qc_retries_twice(self):
        trace, _, _ = self.run_pipeline(faithful=False)
        self.assertEqual(trace["pipeline"]["synthesisAttempts"], 2)
        self.assertTrue(trace["pipeline"]["fallback"])

    def test_qc_unavailable_is_not_a_pass_or_general_fallback(self):
        trace, response, agent = self.run_pipeline(unavailable=True)
        self.assertEqual(trace["pipeline"]["qcStatus"], "error")
        self.assertFalse(trace["pipeline"]["fallback"])
        self.assertEqual(response.answer, "answer")
        agent.synthesis_agent.assert_called_once()
        agent.general_chat_agent.assert_not_called()

    def test_errors_do_not_persist_secrets(self):
        trace = {}
        token = execution_trace.set(trace)
        try: failure("judge", ValueError("secret-key"))
        finally: execution_trace.reset(token)
        self.assertNotIn("secret-key", str(trace))

    def test_json_provider_error_is_recorded_without_leaking_body(self):
        provider_error = RuntimeError('429 {"error":{"message":"secret quota exhausted"}}')
        provider_error.status_code = 429
        trace = {"errors": []}; token = execution_trace.set(trace)
        try:
            with patch("app.services.chat_service.get_client"), patch("app.services.chat_service.generate", side_effect=provider_error):
                with self.assertRaises(RuntimeError):
                    generate_chat_response(ChatRequest(question="question", history=[]))
        finally:
            execution_trace.reset(token)
        self.assertEqual(trace["errors"][0]["stage"], "router")
        self.assertEqual(trace["errors"][0]["httpStatus"], 429)
        self.assertNotIn("secret", str(trace))

    def test_benchmark_diagnostic_redacts_keys_and_preserves_json_braces(self):
        from app.core.config import settings
        from app.core.execution_trace import log_benchmark_failure
        lines = []
        from loguru import logger
        sink = logger.add(lambda message: lines.append(str(message)), format="{message}")
        try:
            with patch.object(settings, "openai_api_key", "private-test-key"):
                log_benchmark_failure("execute", RuntimeError('{"error":"private-test-key"}'))
        finally:
            logger.remove(sink)
        text = "".join(lines)
        self.assertIn('[REDACTED]', text)
        self.assertNotIn('private-test-key', text)

    def test_execute_forbids_ground_truth_and_clears_history(self):
        data = dict(question="q", spaceId=1, documentIds=[1, 2, 3, 4], config={}, corpusFingerprint="hash")
        with self.assertRaises(ValidationError): ExecuteSample(**data, referenceAnswer="LEAK")
        with patch("app.services.benchmark_service.verify_config"), patch("app.services.benchmark_service.prepare_corpus", return_value={"fingerprint": "hash"}), patch("app.services.benchmark_service.generate_chat_response") as generate:
            generate.return_value.model_dump.return_value = {"answer": "a"}
            config = {"chatModel": "c", "routerModel": "r", "evaluatorModel": "e"}
            execute(ExecuteSample(**{**data, "config": config}))
            request = generate.call_args.args[0]
            self.assertEqual(request.history, [])
            self.assertEqual(request.chat_summary, "")
            self.assertEqual(request.document_ids, [1, 2, 3, 4])
            self.assertIsNone(execution_trace.get())

    def test_execute_retains_safe_failure_trace_when_provider_fails(self):
        error = RuntimeError('private provider body')
        error.status_code = 401
        data = dict(question="q", spaceId=1, documentIds=[1], config={}, corpusFingerprint="hash")
        with patch("app.services.benchmark_service.verify_config"), \
                patch("app.services.benchmark_service.prepare_corpus", return_value={"fingerprint": "hash"}), \
                patch("app.services.benchmark_service.generate_chat_response", side_effect=error):
            trace = execute(ExecuteSample(**data))
        self.assertEqual(trace["errors"][0]["httpStatus"], 401)
        self.assertNotIn("private provider body", str(trace))
        self.assertNotIn("response", trace)
        self.assertGreaterEqual(trace["pipeline"]["timings"]["totalMs"], 0)
        self.assertIsNone(execution_trace.get())


class EvaluatorTests(unittest.TestCase):
    def test_benchmark_qc_unavailable_is_recorded_as_error(self):
        from app.services.chat_service import MultiAgentOrchestrator
        error = RuntimeError('Provider {"error":"private"}')
        error.status_code = 503
        trace = {"errors": []}; token = execution_trace.set(trace)
        try:
            with patch("app.services.chat_service.get_client"), patch("app.services.chat_service.generate", side_effect=error):
                result = MultiAgentOrchestrator().evaluator_agent("answer", [{"text": "context"}])
        finally:
            execution_trace.reset(token)
        self.assertEqual(result, (False, None))
        self.assertEqual(trace["errors"][0]["httpStatus"], 503)
        self.assertNotIn("private", str(trace))

    def test_normal_chat_does_not_pass_when_qc_unavailable(self):
        from app.services.chat_service import MultiAgentOrchestrator
        with patch("app.services.chat_service.get_client"), patch("app.services.chat_service.generate", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                MultiAgentOrchestrator().evaluator_agent("answer", [{"text": "context"}])

    def test_incomplete_evaluation_is_not_a_default_pass(self):
        from app.services.chat_service import MultiAgentOrchestrator
        with patch("app.services.chat_service.get_client"), patch("app.services.chat_service.generate", return_value='{}'):
            with self.assertRaises(Exception):
                MultiAgentOrchestrator().evaluator_agent("answer", [{"text": "context"}])


class JudgeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_responses_adapter_preserves_schema_without_wrapper_retries(self):
        from app.services.benchmark_judge import ResponsesJudge
        from pydantic import BaseModel
        class Verdict(BaseModel):
            verdict: int
        client = MagicMock()
        client.responses.parse = AsyncMock(return_value=SimpleNamespace(
            status="completed", output_parsed=Verdict(verdict=1)))
        judge = ResponsesJudge(client, "gpt-5-mini")
        result = await judge.agenerate("Check source", Verdict)
        self.assertEqual(result.verdict, 1)
        client.responses.parse.assert_awaited_once_with(
            model="gpt-5-mini", input="Check source", text_format=Verdict,
            store=False, max_output_tokens=8192)
        client.responses.parse.reset_mock()
        client.responses.parse.side_effect = TimeoutError("provider timed out")
        with self.assertRaises(TimeoutError):
            await judge.agenerate("Check source", Verdict)
        self.assertEqual(client.responses.parse.await_count, 1)

    async def test_incomplete_response_is_an_error_not_a_score(self):
        from app.services.benchmark_judge import ResponsesJudge
        client = MagicMock()
        client.responses.parse = AsyncMock(return_value=SimpleNamespace(status="incomplete", output_parsed=None))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            await ResponsesJudge(client, "gpt-5-mini").agenerate("Check source", MagicMock())


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_ragas_metrics_with_mocked_openai_boundary(self):
        # Exercise the pinned Ragas collection classes and their actual method signatures.
        from app.services.benchmark_judge import ResponsesJudge
        from ragas.embeddings import OpenAIEmbeddings
        from app.core.config import settings
        async def generate(self, prompt, response_model):
            name = response_model.__name__
            payload = {
                "ContextPrecisionOutput": {"reason": "supported", "verdict": 1},
                "StatementGeneratorOutput": {"statements": ["answer"]},
                "NLIStatementOutput": {"statements": [{"statement": "answer", "reason": "supported", "verdict": 1}]},
                "AnswerRelevanceOutput": {"question": "q", "noncommittal": 0},
            }
            return response_model(**payload[name])
        request = ScoreSample(question="q", referenceAnswer="reference", trace={"response": {"answer": "answer"}, "pipeline": {"contexts": [{"text": "context"}]}}, config={"judgeModel": "gpt-5-mini", "judgeEmbeddingModel": "text-embedding-3-small"})
        with patch("app.services.benchmark_service.verify_config"), patch.object(settings, "benchmark_judge_api_key", "test"), patch.object(ResponsesJudge, "agenerate", generate), patch.object(OpenAIEmbeddings, "aembed_text", AsyncMock(return_value=[1., 0.])), patch.object(OpenAIEmbeddings, "aembed_texts", AsyncMock(return_value=[[1., 0.]] * 3)):
            result = await score(request)
        for metric in result.values():
            self.assertEqual(metric["status"], "ok", metric)
            self.assertAlmostEqual(metric["value"], 1.)

    async def test_empty_context_and_judge_failure_are_explicit(self):
        from app.services.benchmark_judge import ResponsesJudge
        from app.core.config import settings
        request = ScoreSample(question="q", referenceAnswer="reference", trace={"response": {"answer": "answer"}, "pipeline": {"contexts": []}}, config={"judgeModel": "gpt-5-mini", "judgeEmbeddingModel": "text-embedding-3-small"})
        with patch("app.services.benchmark_service.verify_config"), patch.object(settings, "benchmark_judge_api_key", "test"), patch.object(ResponsesJudge, "agenerate", AsyncMock(side_effect=RuntimeError("secret"))):
            result = await score(request)
        self.assertEqual(result["contextPrecision"]["value"], 0)
        self.assertEqual(result["faithfulness"]["status"], "not_applicable")
        self.assertEqual(result["answerRelevance"]["status"], "error")
        self.assertIsNone(result["answerRelevance"]["value"])
        self.assertNotIn("secret", str(result))


class ApiTests(unittest.TestCase):
    def test_parse_uses_ai_server_key_and_model(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.v1.endpoints.benchmark import router
        from app.core.config import settings
        app = FastAPI(); app.include_router(router)
        with TestClient(app) as client, patch.object(settings, "openai_api_key", "server-key"), patch("app.services.parser_service.parse_pdf_layout_and_diagrams", return_value=[{"pageNumber": 1, "text": "parsed"}]) as parser:
            response = client.post("/internal/benchmarks/parse", files={"file": ("source.pdf", b"pdf", "application/pdf")}, data={"api_key": "client-key", "parser_model": "client-model"})
            self.assertEqual(response.status_code, 200)
            parser.assert_called_once_with(b"pdf")
            self.assertNotIn("server-key", response.text)

    def test_index_uses_server_key_and_rejects_client_override(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.v1.endpoints.benchmark import router
        from app.core.config import settings
        index = MagicMock(return_value={"status": "success"})
        fake = SimpleNamespace(IndexRequest=lambda **kw: SimpleNamespace(**kw), index_endpoint=index)
        app = FastAPI(); app.include_router(router)
        body = {"documentId": 1, "spaceId": 2, "documentName": "source.pdf", "pages": [{"pageNumber": 1, "text": "parsed"}]}
        with TestClient(app) as client, patch.object(settings, "openai_api_key", "server-key"), patch.dict(sys.modules, {"app.api.v1.endpoints.index": fake}):
            self.assertEqual(client.post("/internal/benchmarks/index", json=body).status_code, 200)
            self.assertFalse(hasattr(index.call_args.args[0], "apiKey"))
            self.assertEqual(client.post("/internal/benchmarks/index", json={**body, "apiKey": "client-key"}).status_code, 422)

    def test_internal_config_needs_no_token(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.v1.endpoints.benchmark import router
        app = FastAPI(); app.include_router(router)
        with TestClient(app) as client, patch("app.api.v1.endpoints.benchmark.configuration", return_value={"chatModel": "fixture"}):
            self.assertEqual(client.get("/internal/benchmarks/config").json(), {"chatModel": "fixture"})


class CorpusPreparationTests(unittest.TestCase):
    def point(self, doc_id):
        return SimpleNamespace(payload={"chunk_id": f"doc-{doc_id}", "document_id": doc_id,
            "page_number": 1, "content": "Source", "enriched_content": "Source"})

    def test_initial_prepare_repairs_only_missing_documents_from_saved_pages(self):
        from app.services.benchmark_service import prepare_corpus, CorpusRequest
        store = MagicMock()
        store.collection_name = "mora_chunks_openai_v1"
        store.client.scroll.side_effect = [([self.point(1)], None), ([self.point(2), self.point(1)], None)]
        with patch.dict(sys.modules, {"app.services.vector_service": SimpleNamespace(vector_store=store)}), \
                patch("app.api.v1.endpoints.index.index_endpoint") as index:
            result = prepare_corpus(CorpusRequest(spaceId=3, documentIds=[1, 2], context=[
                {"documentId": 1, "pageNumber": 1, "text": "Existing source"},
                {"documentId": 2, "documentName": "Missing.pdf", "pageNumber": 2, "text": "Saved page 2"},
                {"documentId": 2, "documentName": "Missing.pdf", "pageNumber": 1, "text": "Saved page 1"}
            ]))
        index.assert_called_once()
        payload = index.call_args.args[0]
        self.assertEqual(payload.documentId, 2)
        self.assertEqual(payload.spaceId, 3)
        self.assertEqual([page.text for page in payload.pages], ["Saved page 1", "Saved page 2"])
        self.assertEqual(result["chunkCount"], 2)
        self.assertTrue(result["fingerprint"])
        for call in store.client.scroll.call_args_list:
            self.assertEqual(call.kwargs["collection_name"], "mora_chunks_openai_v1")

    def test_frozen_run_does_not_reindex_a_missing_corpus(self):
        from app.services.benchmark_service import prepare_corpus, CorpusRequest
        store = MagicMock()
        store.client.scroll.return_value = ([], None)
        with patch.dict(sys.modules, {"app.services.vector_service": SimpleNamespace(vector_store=store)}), \
                patch("app.api.v1.endpoints.index.index_endpoint") as index:
            with self.assertRaisesRegex(ValueError, "document IDs: 2"):
                prepare_corpus(CorpusRequest(spaceId=3, documentIds=[2]))
        index.assert_not_called()

    def test_failed_embedding_stops_preparation_without_a_fingerprint(self):
        from app.services.benchmark_service import prepare_corpus, CorpusRequest
        store = MagicMock()
        store.client.scroll.return_value = ([], None)
        with patch.dict(sys.modules, {"app.services.vector_service": SimpleNamespace(vector_store=store)}), \
                patch("app.api.v1.endpoints.index.index_endpoint", side_effect=RuntimeError("embedding unavailable")):
            with self.assertRaises(RuntimeError):
                prepare_corpus(CorpusRequest(spaceId=3, documentIds=[2], context=[
                    {"documentId": 2, "pageNumber": 1, "text": "Source"}]))
        self.assertEqual(store.client.scroll.call_count, 1)


class RetrievalTests(unittest.TestCase):
    def test_rerank_failure_is_reported_and_preserves_rrf_fallback(self):
        store = MagicMock()
        store.search_dense.return_value = [dict(chunkId="a", content="alpha", enrichedContent="alpha")]
        store.search_sparse.return_value = [dict(chunkId="b", content="beta", enrichedContent="beta")]
        fake = SimpleNamespace(vector_store=store)
        # Avoid constructing a real Qdrant client just to test retrieval ordering.
        with patch.dict(sys.modules, {"app.services.vector_service": fake}):
            retrieval = importlib.import_module("app.services.retrieval_service")
        client = MagicMock()
        ranker = MagicMock(); ranker.rerank.side_effect = RuntimeError("ranker unavailable")
        trace = {}; token = execution_trace.set(trace)
        try:
            with patch.object(retrieval, "vector_store", store), patch.object(retrieval, "get_ranker", return_value=ranker), patch.object(retrieval, "embed", return_value=[[1., 0.]]), patch.object(retrieval.settings, "retrieval_enabled", True):
                result = retrieval.hybrid_retrieve("q", 1, client, document_ids=[1, 2])
        finally:
            execution_trace.reset(token)
        self.assertEqual([c["chunkId"] for c in result], ["a", "b"])
        self.assertEqual(trace["retrieval"]["rerank"], "unavailable_or_failed")
        self.assertEqual(trace["retrieval"]["mode"], "hybrid")
        store.search_dense.assert_called_once()
        store.search_sparse.assert_called_once()


if __name__ == "__main__":
    unittest.main()
