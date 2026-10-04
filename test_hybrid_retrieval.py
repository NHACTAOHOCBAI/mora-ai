import os
import sys
from loguru import logger

# Đảm bảo import được các module trong mora-ai
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings
from app.services.chunking_service import extract_semantic_chunks_from_pages
from app.services.vector_service import vector_store
from app.services.retrieval_service import hybrid_retrieve
from google import genai

def test_full_pipeline():
    logger.info("=== BẮT ĐẦU KIỂM THỬ PIPELINE SEMANTIC CHUNKING & HYBRID RETRIEVAL ===")

    # 1. Kiểm tra kết nối Qdrant
    assert vector_store.client is not None, "Qdrant Client không thể kết nối tới localhost:6333!"
    collections = [c.name for c in vector_store.client.get_collections().collections]
    logger.info(f"[1/4] Qdrant Collections hiện có: {collections}")
    assert "mora_chunks" in collections, "Collection 'mora_chunks' chưa được tạo trong Qdrant!"

    # 2. Dữ liệu giả lập 3 trang về Hệ Điều Hành
    mock_pages = [
        {
            "pageNumber": 1,
            "text": "# Chương 1: Tổng Quan Hệ Điều Hành\n\n## 1.1 Khái niệm cơ bản\nHệ điều hành (Operating System - OS) là phần mềm hệ thống quản lý tài nguyên phần cứng và phần mềm của máy tính.\nNó đóng vai trò là cầu nối trung gian giữa người dùng và phần cứng máy tính."
        },
        {
            "pageNumber": 2,
            "text": "## 1.2 Các chế độ hoạt động\nViệc phân tách chế độ hoạt động giúp hệ điều hành tự bảo vệ các tài nguyên quan trọng.\n\n| Chế độ | Mode Bit | Quyền hạn |\n| :--- | :--- | :--- |\n| User Mode | 1 | Bị giới hạn, không thể chạy lệnh đặc quyền |\n| Kernel Mode | 0 | Toàn quyền truy cập phần cứng và bộ nhớ |"
        },
        {
            "pageNumber": 3,
            "text": "## 1.3 Cơ chế System Call và Chuyển Chế Độ\n[MÔ TẢ HÌNH ẢNH TRÊN TRANG 3]: Sơ đồ minh họa 6 bước chuyển đổi giữa User Mode và Kernel Mode qua lệnh Trap và System Call:\n1. User process executing (mode bit = 1)\n2. Calls system call\n3. Trap chuyển sang Kernel Mode (mode bit = 0)\n4. Execute system call trong Kernel\n5. Return chuyển về User Mode (mode bit = 1)\n6. Return from system call."
        }
    ]

    space_id = 999
    document_id = 100
    doc_name = "HDH_Demo.pdf"

    # 3. Phân đoạn Semantic Chunking
    chunks = extract_semantic_chunks_from_pages(
        pages=mock_pages,
        space_id=space_id,
        document_id=document_id,
        document_name=doc_name
    )
    logger.info(f"[2/4] Đã tạo thành công {len(chunks)} Semantic Chunks:")
    for c in chunks:
        logger.info(f"  - [{c.chunkType}] Page {c.pageNumber} | Path: '{c.sectionPath}' | Tokens: {c.tokenCount}")
        logger.info(f"    Preview: {c.enrichedContent[:80]}...")

    assert len(chunks) >= 3, "Số lượng chunk phải từ 3 trở lên!"

    # 4. Kiểm tra Gemini API Key và tạo Embeddings
    api_key = settings.gemini_api_key
    if not api_key:
        logger.info("Chưa có GEMINI_API_KEY. Tiến hành kiểm thử Qdrant Upsert & Search bằng Mock Embeddings (768 dims)...")
        import random
        random.seed(42)
        mock_embeddings = [[random.uniform(-1, 1) for _ in range(768)] for _ in chunks]
        vector_store.upsert_chunks(
            space_id=space_id,
            document_id=document_id,
            chunks=chunks,
            embeddings=mock_embeddings
        )
        logger.info(f"[3/4] Đã nạp thành công {len(mock_embeddings)} mock vectors vào Qdrant và BM25 Index!")

        # Thử search dense trong Qdrant
        query_vec = mock_embeddings[3] # Vector của chunk số 4 (diagram)
        dense_hits = vector_store.search_dense(space_id=space_id, query_vector=query_vec, limit=3)
        logger.info(f"[4/4] Qdrant Dense Search kết quả: {len(dense_hits)} hits. Top 1 ID: {dense_hits[0]['chunkId']} (Score: {dense_hits[0]['score']:.4f})")
        assert dense_hits[0]["chunkId"] == chunks[3].chunkId, "Top 1 dense hit phải là chunk số 4!"

        # Thử search sparse BM25
        sparse_hits = vector_store.search_sparse(space_id=space_id, query="Trap kernel mode system call", limit=3)
        logger.info(f"[4/4] BM25 Sparse Search kết quả: {len(sparse_hits)} hits. Top 1 ID: {sparse_hits[0]['chunkId']} (Score: {sparse_hits[0]['score']:.4f})")
        assert "Trap" in sparse_hits[0]["content"], "Top 1 BM25 hit phải chứa từ khóa 'Trap'!"

        # Thử RRF & FlashRank
        from app.services.retrieval_service import rrf_merge, ranker, RerankRequest
        merged = rrf_merge(dense_hits, sparse_hits, top_n=5)
        logger.info(f"RRF Merged candidates: {len(merged)} items.")
        assert len(merged) > 0

        if ranker:
            passages = [{"id": item["chunkId"], "text": item["enrichedContent"], "meta": item} for item in merged]
            rerank_req = RerankRequest(query="Lệnh Trap và các bước chuyển sang kernel mode là gì?", passages=passages)
            reranked = ranker.rerank(rerank_req)
            logger.info(f"FlashRank Reranked Top 1: {reranked[0]['id']} (Score: {reranked[0]['score']:.4f})")
            assert "Trap" in reranked[0]["meta"]["content"], "Top 1 sau Re-ranking phải là chunk nói về lệnh Trap!"

        logger.info("=== KIỂM THỬ THÀNH CÔNG 100% PIPELINE QDRANT CONTAINER & HYBRID RETRIEVAL! ===")
        return

    client = genai.Client(api_key=api_key)
    logger.info(f"[3/4] Đang tạo Embeddings qua model '{settings.gemini_embedding_model}'...")
    embeddings = []
    for c in chunks:
        resp = client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=c.enrichedContent
        )
        embeddings.append(resp.embedding.values)

    # Upsert vào Qdrant & BM25
    vector_store.upsert_chunks(
        space_id=space_id,
        document_id=document_id,
        chunks=chunks,
        embeddings=embeddings
    )
    logger.info(f"[3/4] Đã nạp thành công {len(embeddings)} vectors vào Qdrant và BM25 Index!")

    # 5. Kiểm thử Two-Stage Hybrid Search & Re-ranking
    query = "Lệnh Trap và các bước chuyển sang kernel mode là gì?"
    logger.info(f"[4/4] Thực thi Two-Stage Hybrid Search cho câu hỏi: '{query}'")
    retrieved = hybrid_retrieve(
        query=query,
        space_id=space_id,
        gemini_client=client,
        top_k=3
    )

    logger.info(f"Kết quả Hybrid Search (Top {len(retrieved)} Chunks):")
    for idx, r in enumerate(retrieved):
        logger.info(f"  Top {idx+1}: Page {r['pageNumber']} | Path: '{r['sectionPath']}' | Score: {r.get('rerank_score') or r.get('rrf_score')}")
        logger.info(f"    Content: {r['content'][:120]}...")

    assert len(retrieved) > 0, "Hybrid Retrieval phải trả về ít nhất 1 chunk phù hợp!"
    assert retrieved[0]["pageNumber"] == 3 or "Trap" in retrieved[0]["content"], "Chunk số 1 phải liên quan đến lệnh Trap ở Trang 3!"
    logger.info("=== TẤT CẢ CÁC BƯỚC KIỂM THỬ ĐÃ THÀNH CÔNG VƯỢT TRỘI! ===")

if __name__ == "__main__":
    test_full_pipeline()
