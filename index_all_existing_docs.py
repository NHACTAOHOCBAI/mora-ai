import os
import sys
import json
import psycopg2
from loguru import logger
from google import genai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings
from app.services.chunking_service import extract_semantic_chunks_from_pages
from app.services.vector_service import vector_store

def index_all():
    logger.info("Đang quét toàn bộ tài liệu từ PostgreSQL (port 5433) để nạp vào Qdrant...")
    conn = psycopg2.connect(
        dbname="mora_db",
        user="postgres",
        password="postgres",
        host="localhost",
        port=5433
    )
    cur = conn.cursor()
    
    # Lấy danh sách tài liệu
    cur.execute("SELECT id, name, space_id FROM documents WHERE status = 'READY';")
    docs = cur.fetchall()
    logger.info(f"Tìm thấy {len(docs)} tài liệu đã sẵn sàng trong DB.")

    # Lấy API Key từ user_ai_settings
    cur.execute("SELECT gemini_api_key FROM user_ai_settings WHERE gemini_api_key IS NOT NULL AND gemini_api_key != '' LIMIT 1;")
    key_row = cur.fetchone()
    api_key = key_row[0] if key_row else settings.gemini_api_key

    if not api_key:
        logger.error("Không tìm thấy Gemini API Key trong DB hoặc file .env.")
        return

    client = genai.Client(api_key=api_key)

    for doc_id, doc_name, space_id in docs:
        cur.execute("SELECT page_number, text FROM document_pages WHERE document_id = %s ORDER BY page_number ASC;", (doc_id,))
        pages_raw = cur.fetchall()
        pages = [{"pageNumber": r[0], "text": r[1]} for r in pages_raw]
        
        logger.info(f"Đang bóc tách Semantic Chunks cho Document '{doc_name}' (ID: {doc_id}, Space: {space_id}, Pages: {len(pages)})...")
        chunks = extract_semantic_chunks_from_pages(
            pages=pages,
            space_id=space_id,
            document_id=doc_id,
            document_name=doc_name
        )
        
        if not chunks:
            continue
            
        logger.info(f"Đang tạo {len(chunks)} Embeddings với model '{settings.gemini_embedding_model}'...")
        embeddings = []
        import time
        for idx, c in enumerate(chunks):
            for attempt in range(5):
                try:
                    resp = client.models.embed_content(
                        model=settings.gemini_embedding_model,
                        contents=c.enrichedContent
                    )
                    val = resp.embeddings[0].values if resp.embeddings else []
                    embeddings.append(val)
                    time.sleep(0.05)
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        logger.warning(f"Rate limit hit at chunk {idx+1}/{len(chunks)}. Sleeping 10s...")
                        time.sleep(10)
                    else:
                        raise e
            
        vector_store.upsert_chunks(
            space_id=space_id,
            document_id=doc_id,
            chunks=chunks,
            embeddings=embeddings
        )
        logger.info(f"==> Đã nạp thành công Document ID {doc_id} vào Qdrant cho Space {space_id}!")

    cur.close()
    conn.close()
    logger.info("=== HOÀN TẤT ĐỒNG BỘ TOÀN BỘ TÀI LIỆU VÀO QDRANT VECTOR STORE! ===")

if __name__ == "__main__":
    index_all()
