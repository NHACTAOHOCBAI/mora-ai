import io
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pypdf
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Import the app without touching a running database.
memory = QdrantClient(":memory:")
with patch("qdrant_client.QdrantClient", return_value=memory):
    from app.main import app
    from app.services.vector_service import vector_store

from app.core.config import settings
from app.schemas.chunk import SemanticChunk
from app.services import openai_service, chat_service, parser_service


class OpenAiMigrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_missing_server_key_is_service_error_and_settings_endpoints_are_removed(self):
        with patch.object(settings, "openai_api_key", None):
            response = self.client.post("/api/chat", json={"question": "Xin chào", "history": []})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("API Key", response.json()["detail"])
        self.assertEqual(self.client.post("/api/chat/validate-key", json={}).status_code, 404)
        for name in ("ChatRequest", "ChatSummarizeRequest", "IndexRequest"):
            properties = app.openapi()["components"]["schemas"][name]["properties"]
            self.assertFalse(set(properties) & {"api_key", "apiKey", "chat_model", "summarizer_model"})

    def test_server_key_timeout_retry_and_no_credentials_from_request(self):
        with patch.object(settings, "openai_api_key", "server-owned"), patch.object(openai_service, "OpenAI") as sdk:
            openai_service.get_client()
        self.assertEqual(sdk.call_args.kwargs["api_key"], "server-owned")
        self.assertEqual(sdk.call_args.kwargs["timeout"], 60)
        self.assertEqual(sdk.call_args.kwargs["max_retries"], 2)

    def test_general_chat_and_provider_failures(self):
        sdk = Mock()
        sdk.responses.parse.return_value = SimpleNamespace(
            status="completed", output_parsed=chat_service.RouteSchema(intent="GENERAL", reason="greeting"))
        sdk.responses.create.return_value = SimpleNamespace(status="completed", output_text="Xin chào bạn!")
        with patch.object(chat_service, "get_client", return_value=sdk):
            response = self.client.post("/api/chat", json={"question": "Chào", "history": []})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["answer"], "Xin chào bạn!")
            self.assertFalse(sdk.responses.create.call_args.kwargs["store"])
            for failure in ("401 invalid key secret-value", "429 quota secret-value", "timeout secret-value"):
                sdk.responses.create.side_effect = RuntimeError(failure)
                response = self.client.post("/api/chat", json={"question": "Chào", "history": []})
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("secret-value", response.text)

    def test_rag_citations_and_summary_preserve_contract(self):
        citation = {"pageNumber": 2, "quote": "Nguồn", "documentId": 7, "documentName": "Bài học"}
        with patch.object(chat_service, "get_client", return_value=Mock()), patch.object(chat_service, "generate") as generate:
            generate.side_effect = [
                json.dumps({"intent": "RAG", "reason": "learning"}),
                json.dumps({"answer": "Nội dung", "citations": [citation, citation],
                            "condensedQuestion": "Hỏi", "hasSufficientInfo": True}),
                json.dumps({"is_faithful": True, "hallucinations": [], "score": 1.0})
            ]
            response = self.client.post("/api/chat", json={"question": "Hỏi", "history": [],
                "context": [{"pageNumber": 2, "text": "Nguồn", "documentId": 7}]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["citations"], [citation])
            self.assertTrue(response.json()["promptSent"])
            generate.side_effect = None
            generate.return_value = "Tóm tắt"
            response = self.client.post("/api/chat/summarize", json={"history": [{"sender": "user", "text": "Hỏi"}]})
            self.assertEqual(response.json(), {"summary": "Tóm tắt"})

    def test_pdf_pages_sent_as_pdf_and_markdown_retained(self):
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=300, height=300)
        writer.add_blank_page(width=300, height=300)
        stream = io.BytesIO()
        writer.write(stream)
        markdown = "# Bảng\n| A | B |\n|---|---|\n| 1 | 2 |\n[MÔ TẢ HÌNH]: Biểu đồ"
        sdk = Mock()
        sdk.responses.create.return_value = SimpleNamespace(status="completed", output_text=markdown)
        with patch.object(parser_service, "get_client", return_value=sdk):
            response = self.client.post("/api/parse", files={"file": ("lesson.pdf", stream.getvalue(), "application/pdf")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([p["pageNumber"] for p in response.json()["pages"]], [1, 2])
        self.assertEqual(response.json()["pages"][0]["text"], markdown)
        content = sdk.responses.create.call_args.kwargs["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_file")
        self.assertTrue(content[0]["file_data"].startswith("data:application/pdf;base64,"))

    def test_embedding_failures_never_upsert_partial_index(self):
        from app.api.v1.endpoints import index
        with patch.object(index, "get_client", return_value=Mock()), \
                patch.object(index, "embed", side_effect=RuntimeError("quota")), \
                patch.object(index.vector_store, "upsert_chunks") as upsert:
            response = self.client.post("/api/index", json={"documentId": 7, "spaceId": 1,
                "documentName": "Lesson", "pages": [{"pageNumber": 1, "text": "# Title\nUseful learning content."}]})
        self.assertEqual(response.status_code, 503)
        upsert.assert_not_called()

    def test_embedding_rejects_wrong_dimension_or_missing_items(self):
        sdk = Mock()
        for items in ([], [SimpleNamespace(index=0, embedding=[0.0] * 12)]):
            sdk.embeddings.create.return_value = SimpleNamespace(data=items)
            with self.assertRaises(RuntimeError):
                openai_service.embed(sdk, ["hello"])

    def test_reindex_is_idempotent_and_does_not_touch_old_collection(self):
        old = "mora_chunks"
        if not memory.collection_exists(old):
            memory.create_collection(old, vectors_config=VectorParams(size=3072, distance=Distance.COSINE))
        def chunk(name):
            return SemanticChunk(chunkId=name, documentId=17, spaceId=18, pageNumber=2,
                sectionPath="", chunkType="TEXT", content="test content", enrichedContent="test content", tokenCount=2)
        vector = [0.1] * 1536
        vector_store.upsert_chunks(18, 17, [chunk("a"), chunk("b")], [vector, vector])
        vector_store.upsert_chunks(18, 17, [chunk("a"), chunk("b")], [vector, vector])
        self.assertEqual(memory.count(vector_store.collection_name).count, 2)
        vector_store.upsert_chunks(18, 17, [chunk("a")], [vector])
        self.assertEqual(memory.count(vector_store.collection_name).count, 1)
        self.assertEqual(memory.get_collection(old).config.params.vectors.size, 3072)
        with self.assertRaises(ValueError):
            vector_store.upsert_chunks(18, 17, [chunk("a")], [[0.1] * 20])
        vector_store.bm25_store.clear()
        vector_store.search_sparse(18, "test")
        self.assertEqual(len(vector_store.bm25_store[18]["chunks"]), 1)

    def test_mismatched_collection_is_not_deleted(self):
        sdk = Mock()
        sdk.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name=vector_store.collection_name)])
        sdk.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=3072))))
        with patch.object(vector_store, "client", sdk), self.assertRaises(ValueError):
            vector_store._ensure_collection()
        sdk.delete_collection.assert_not_called()

    def test_retrieval_cannot_start_before_migration_is_enabled(self):
        from app.services.retrieval_service import hybrid_retrieve
        with patch.object(settings, "retrieval_enabled", False), self.assertRaises(RuntimeError):
            hybrid_retrieve("question", 1, Mock())

    def test_chat_uses_saved_pages_while_vector_migration_is_disabled(self):
        context = [{"pageNumber": 2, "text": "Saved lesson", "documentId": 7}]
        with patch.object(settings, "retrieval_enabled", False), \
                patch.object(chat_service, "get_client", return_value=Mock()), \
                patch("app.services.retrieval_service.hybrid_retrieve") as retrieve:
            result = chat_service.MultiAgentOrchestrator().retrieval_agent(
                "question", space_id=1, raw_context=context, document_ids=[7])
        self.assertEqual(result, context)
        retrieve.assert_not_called()

    def test_empty_allowed_document_list_never_searches_stale_vectors(self):
        with patch.object(settings, "retrieval_enabled", True), \
                patch.object(chat_service, "get_client", return_value=Mock()), \
                patch.object(chat_service, "generate") as generate, \
                patch("app.services.retrieval_service.hybrid_retrieve") as retrieve:
            generate.side_effect = [json.dumps({"intent": "RAG", "reason": "learning"}), "General answer"]
            response = self.client.post("/api/chat", json={"question": "Question", "history": [],
                                       "space_id": 1, "document_ids": [], "context": []})
        self.assertEqual(response.status_code, 200)
        retrieve.assert_not_called()
        self.assertEqual(response.json()["citations"], [])

    def test_slow_pdf_parsing_does_not_block_other_requests(self):
        started, release = Event(), Event()
        def slow_parse(content):
            started.set()
            release.wait(timeout=5)
            return [{"pageNumber": 1, "text": "Parsed"}]
        with patch("app.api.v1.endpoints.parse.parse_pdf_layout_and_diagrams", side_effect=slow_parse), \
                TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
            parse = pool.submit(client.post, "/api/parse", files={"file": ("lesson.pdf", b"pdf", "application/pdf")})
            try:
                self.assertTrue(started.wait(timeout=3))
                hello = pool.submit(client.get, "/api/hello")
                self.assertEqual(hello.result(timeout=2).status_code, 200)
            finally:
                release.set()
            self.assertEqual(parse.result(timeout=3).status_code, 200)


if __name__ == "__main__":
    unittest.main()
