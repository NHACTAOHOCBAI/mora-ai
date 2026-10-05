import uuid
from typing import List, Dict, Any, Optional
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    PointStruct, 
    Filter, 
    FieldCondition, 
    MatchValue,
    MatchAny,
    PayloadSchemaType
)
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.schemas.chunk import SemanticChunk

class VectorStoreManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.collection_name = "mora_chunks"
        self.vector_size = 3072  # Kích thước embedding chuẩn của models/gemini-embedding-2
        
        try:
            logger.info(f"Kết nối tới Qdrant Vector Database tại {settings.qdrant_host}:{settings.qdrant_port}...")
            self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
            self._ensure_collection()
            logger.info("Kết nối Qdrant Vector DB thành công!")
        except Exception as e:
            logger.error(f"Lỗi khi kết nối tới Qdrant: {e}", exc_info=True)
            self.client = None

        # BM25 Storage: Dictionary lưu trữ index BM25 và chunks theo space_id
        # { space_id: { "bm25": BM25Okapi, "chunks": [SemanticChunk, ...] } }
        self.bm25_store: Dict[int, Dict[str, Any]] = {}
        self._initialized = True

    def _ensure_collection(self):
        if not self.client:
            return
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            # Kiểm tra kích thước vector hiện tại, nếu khác thì xóa tạo lại
            info = self.client.get_collection(self.collection_name)
            curr_size = info.config.params.vectors.size
            if curr_size != self.vector_size:
                logger.info(f"Kích thước collection ({curr_size}) khác {self.vector_size}. Đang xóa và tạo lại collection '{self.collection_name}'...")
                self.client.delete_collection(self.collection_name)
                collections.remove(self.collection_name)

        if self.collection_name not in collections:
            logger.info(f"Đang khởi tạo Qdrant Collection '{self.collection_name}' (size={self.vector_size}, distance=Cosine)...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            # Tạo payload index để filter theo space_id và document_id siêu tốc
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="space_id",
                field_schema=PayloadSchemaType.INTEGER
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=PayloadSchemaType.INTEGER
            )
            logger.info(f"Khởi tạo collection '{self.collection_name}' và payload index thành công!")

    def upsert_chunks(
        self, 
        space_id: int, 
        document_id: int, 
        chunks: List[SemanticChunk], 
        embeddings: List[List[float]]
    ):
        """Nạp các chunks và vector tương ứng vào Qdrant và cập nhật BM25 Index."""
        if not self.client:
            raise RuntimeError("Qdrant Client chưa sẵn sàng.")

        if len(chunks) != len(embeddings):
            raise ValueError(f"Số lượng chunks ({len(chunks)}) không khớp với số lượng embeddings ({len(embeddings)})")

        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            # Đảm bảo vector đúng kích thước
            if len(vector) > self.vector_size:
                vector = vector[:self.vector_size]
            elif len(vector) < self.vector_size:
                vector = vector + [0.0] * (self.vector_size - len(vector))

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.chunkId}_{space_id}_{document_id}"))
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "chunk_id": chunk.chunkId,
                    "space_id": space_id,
                    "document_id": document_id,
                    "document_name": chunk.documentName or "",
                    "page_number": chunk.pageNumber,
                    "section_path": chunk.sectionPath,
                    "chunk_type": chunk.chunkType,
                    "content": chunk.content,
                    "enriched_content": chunk.enrichedContent,
                    "token_count": chunk.tokenCount
                }
            ))

        # 1. Upsert vào Qdrant
        logger.info(f"[Qdrant] Đang nạp {len(points)} vector points cho Space {space_id}, Document {document_id}...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"[Qdrant] Nạp {len(points)} points thành công!")

        # 2. Cập nhật BM25 In-memory Index cho Space
        self._update_bm25(space_id, chunks)

    def _update_bm25(self, space_id: int, new_chunks: List[SemanticChunk]):
        """Cập nhật chỉ mục BM25 cho một space."""
        if space_id not in self.bm25_store:
            self.bm25_store[space_id] = {
                "chunks": [],
                "bm25": None
            }

        # Loại bỏ các chunk cũ cùng document_id nếu có
        existing_chunks = self.bm25_store[space_id]["chunks"]
        doc_ids = {c.documentId for c in new_chunks}
        filtered_chunks = [c for c in existing_chunks if c.documentId not in doc_ids]
        filtered_chunks.extend(new_chunks)

        self.bm25_store[space_id]["chunks"] = filtered_chunks

        # Tokenize corpus cho BM25
        tokenized_corpus = [c.enrichedContent.lower().split() for c in filtered_chunks]
        if tokenized_corpus:
            self.bm25_store[space_id]["bm25"] = BM25Okapi(tokenized_corpus)
            logger.info(f"[BM25] Đã cập nhật chỉ mục BM25 cho Space {space_id} với tổng số {len(filtered_chunks)} chunks.")

    def search_dense(
        self, 
        space_id: int, 
        query_vector: List[float], 
        limit: int = 20,
        document_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm tương đồng Vector trong Qdrant lọc theo space_id và danh sách document_ids tùy chọn."""
        if not self.client:
            return []

        if len(query_vector) > self.vector_size:
            query_vector = query_vector[:self.vector_size]
        elif len(query_vector) < self.vector_size:
            query_vector = query_vector + [0.0] * (self.vector_size - len(query_vector))

        must_conditions = [
            FieldCondition(
                key="space_id",
                match=MatchValue(value=space_id)
            )
        ]

        if document_ids and len(document_ids) > 0:
            must_conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=document_ids)
                )
            )

        query_filter = Filter(must=must_conditions)

        search_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit
        )
        search_result = search_response.points

        results = []
        for hit in search_result:
            results.append({
                "chunkId": hit.payload.get("chunk_id"),
                "documentId": hit.payload.get("document_id"),
                "documentName": hit.payload.get("document_name"),
                "pageNumber": hit.payload.get("page_number"),
                "sectionPath": hit.payload.get("section_path"),
                "chunkType": hit.payload.get("chunk_type"),
                "content": hit.payload.get("content"),
                "enrichedContent": hit.payload.get("enriched_content"),
                "score": hit.score,
                "source": "dense"
            })
        return results

    def search_sparse(
        self, 
        space_id: int, 
        query: str, 
        limit: int = 20,
        document_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm từ khóa chính xác BM25 lọc theo space_id và danh sách document_ids tùy chọn."""
        if space_id not in self.bm25_store or not self.bm25_store[space_id]["bm25"]:
            return []

        bm25 = self.bm25_store[space_id]["bm25"]
        chunks = self.bm25_store[space_id]["chunks"]
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Lấy top k điểm cao nhất
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        doc_id_set = set(document_ids) if document_ids and len(document_ids) > 0 else None

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                c = chunks[idx]
                if doc_id_set is not None and c.documentId not in doc_id_set:
                    continue
                results.append({
                    "chunkId": c.chunkId,
                    "documentId": c.documentId,
                    "documentName": c.documentName,
                    "pageNumber": c.pageNumber,
                    "sectionPath": c.sectionPath,
                    "chunkType": c.chunkType,
                    "content": c.content,
                    "enrichedContent": c.enrichedContent,
                    "score": float(scores[idx]),
                    "source": "sparse"
                })
                if len(results) >= limit:
                    break
        return results

vector_store = VectorStoreManager()
