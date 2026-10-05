from typing import List, Dict, Any, Optional
from loguru import logger
from google import genai
from app.core.config import settings
from app.services.vector_service import vector_store

_ranker = None

def get_ranker():
    global _ranker
    if _ranker is None:
        try:
            from flashrank import Ranker
            _ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
            logger.info("Khởi tạo FlashRank Re-ranker (ms-marco-TinyBERT-L-2-v2) thành công!")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo FlashRank: {e}. Sẽ sử dụng RRF scoring dự phòng.")
            _ranker = False
    return _ranker if _ranker is not False else None


def rrf_merge(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60,
    top_n: int = 20
) -> List[Dict[str, Any]]:
    """Hợp nhất kết quả Dense và Sparse bằng Reciprocal Rank Fusion (RRF)."""
    scores: Dict[str, float] = {}
    item_map: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(dense_results):
        chunk_id = item["chunkId"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))
        item_map[chunk_id] = item

    for rank, item in enumerate(sparse_results):
        chunk_id = item["chunkId"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))
        if chunk_id not in item_map:
            item_map[chunk_id] = item

    # Sắp xếp theo điểm RRF giảm dần
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_n]
    merged = []
    for cid in sorted_ids:
        item = item_map[cid]
        item["rrf_score"] = scores[cid]
        merged.append(item)
    return merged

def hybrid_retrieve(
    query: str, 
    space_id: int, 
    gemini_client: genai.Client, 
    top_k: int = None,
    document_ids: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Quy trình Two-Stage Hybrid Search & Re-ranking:
    1. Tạo Dense Vector cho câu hỏi qua Gemini Embedding Model.
    2. Tìm kiếm đồng thời Dense (Qdrant) và Sparse (BM25), có thể lọc theo document_ids.
    3. Hợp nhất bằng Reciprocal Rank Fusion (RRF).
    4. Re-rank bằng FlashRank để chọn ra Top-K chunks tối ưu nhất.
    """
    target_top_k = top_k or settings.retrieval_top_k
    logger.info(f"[Retrieval Agent] Bắt đầu Two-Stage Hybrid Search cho Space {space_id} (docs: {document_ids}) với câu hỏi: '{query}'")

    # 1. Tạo Query Vector
    try:
        embed_resp = gemini_client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=query
        )
        query_vector = embed_resp.embeddings[0].values if embed_resp.embeddings else []
    except Exception as e:
        logger.error(f"[Retrieval Agent] Lỗi khi tạo Query Embedding: {e}", exc_info=True)
        query_vector = []

    # 2. Dense Search (Qdrant) & Sparse Search (BM25)
    dense_hits = []
    if query_vector:
        dense_hits = vector_store.search_dense(
            space_id=space_id, 
            query_vector=query_vector, 
            limit=settings.retrieval_dense_limit,
            document_ids=document_ids
        )

    sparse_hits = vector_store.search_sparse(
        space_id=space_id, 
        query=query, 
        limit=settings.retrieval_sparse_limit,
        document_ids=document_ids
    )

    logger.info(f"[Retrieval Agent] Tìm thấy {len(dense_hits)} Dense hits và {len(sparse_hits)} Sparse hits.")

    # 3. RRF Fusion
    merged_candidates = rrf_merge(dense_hits, sparse_hits, top_n=20)
    if not merged_candidates:
        logger.warn(f"[Retrieval Agent] Không tìm thấy chunk nào trong Vector Store cho Space {space_id}.")
        return []

    # 4. Stage 2: Re-ranking với FlashRank
    active_ranker = get_ranker()
    if active_ranker and len(merged_candidates) > 1:
        try:
            from flashrank import RerankRequest
            passages = [
                {
                    "id": item["chunkId"],
                    "text": item.get("enrichedContent") or item.get("content", ""),
                    "meta": item
                }
                for item in merged_candidates
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            rerank_results = active_ranker.rerank(rerank_request)

            final_chunks = []
            for hit in rerank_results[:target_top_k]:
                meta = hit["meta"]
                meta["rerank_score"] = float(hit["score"])
                final_chunks.append(meta)

            logger.info(f"[Retrieval Agent] FlashRank Re-ranker hoàn tất. Đã chọn {len(final_chunks)} chunks tinh hoa nhất.")
            return final_chunks
        except Exception as e:
            logger.error(f"[Retrieval Agent] Lỗi trong quá trình FlashRank re-ranking: {e}. Sử dụng kết quả RRF.", exc_info=True)

    final_chunks = merged_candidates[:target_top_k]
    logger.info(f"[Retrieval Agent] Đã chọn Top {len(final_chunks)} chunks theo điểm RRF.")
    return final_chunks
