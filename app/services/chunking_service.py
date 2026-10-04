import re
from typing import List
from loguru import logger
from app.schemas.chunk import SemanticChunk

def count_tokens_approx(text: str) -> int:
    """Đếm xấp xỉ số lượng token dựa trên từ và ký tự tiếng Việt / code."""
    return len(text.split())

def extract_semantic_chunks_from_pages(
    pages: List[dict], 
    space_id: int, 
    document_id: int, 
    document_name: str = "",
    max_tokens: int = 500,
    overlap_tokens: int = 80
) -> List[SemanticChunk]:
    """
    Phân đoạn tài liệu thành các Semantic Chunks có nhận thức cấu trúc:
    - Bóc tách phân cấp tiêu đề Markdown (#, ##, ###) tạo Breadcrumb (`sectionPath`).
    - Bảo toàn toàn vẹn bảng biểu (Markdown Tables) và khối mô tả sơ đồ/hình ảnh.
    - Đính kèm Breadcrumb vào `enrichedContent` để tối ưu vector similarity.
    """
    logger.info(f"Bắt đầu phân đoạn Semantic Chunks cho tài liệu '{document_name}' (ID: {document_id}, Space: {space_id}) từ {len(pages)} trang.")
    chunks: List[SemanticChunk] = []
    current_headers: List[str] = []
    chunk_counter = 1

    header_regex = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    table_regex = re.compile(r'(\|.+\|\n\|[-:\s|]+\|\n(?:\|.+\|\n?)+)', re.MULTILINE)
    diagram_regex = re.compile(r'(\[MÔ TẢ HÌNH ẢNH TRÊN TRANG \d+\]:[\s\S]+?)(?=\n#{1,6}\s+|\n\[MÔ TẢ HÌNH ẢNH|\Z)', re.MULTILINE)

    for p in pages:
        page_num = p.get("pageNumber", 1)
        raw_text = (p.get("text") or "").strip()
        if not raw_text:
            continue

        # Phân tách theo dòng để quét cấu trúc
        lines = raw_text.splitlines()
        current_buffer = []
        current_buffer_tokens = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            trimmed = line.strip()

            # 1. Phát hiện Tiêu đề Markdown
            header_match = header_regex.match(trimmed)
            if header_match:
                # Đóng gói buffer hiện tại nếu có
                if current_buffer:
                    chunk_text = "\n".join(current_buffer).strip()
                    if chunk_text:
                        breadcrumb = " > ".join(current_headers) if current_headers else (document_name or f"Tài liệu #{document_id}")
                        enriched = f"[{breadcrumb}]\n{chunk_text}"
                        chunks.append(SemanticChunk(
                            chunkId=f"doc_{document_id}_p{page_num}_chk_{chunk_counter}",
                            documentId=document_id,
                            spaceId=space_id,
                            documentName=document_name,
                            pageNumber=page_num,
                            sectionPath=breadcrumb,
                            chunkType="TEXT",
                            content=chunk_text,
                            enrichedContent=enriched,
                            tokenCount=current_buffer_tokens
                        ))
                        chunk_counter += 1
                    current_buffer = []
                    current_buffer_tokens = 0

                level = len(header_match.group(1))
                header_title = header_match.group(2).strip()

                # Cắt tỉa stack header theo cấp độ
                if level <= len(current_headers):
                    current_headers = current_headers[:level - 1]
                current_headers.append(header_title)
                i += 1
                continue

            # 2. Phát hiện Khối mô tả hình ảnh / Sơ đồ
            if trimmed.startswith("[MÔ TẢ HÌNH ẢNH TRÊN TRANG"):
                if current_buffer:
                    chunk_text = "\n".join(current_buffer).strip()
                    if chunk_text:
                        breadcrumb = " > ".join(current_headers) if current_headers else (document_name or f"Tài liệu #{document_id}")
                        enriched = f"[{breadcrumb}]\n{chunk_text}"
                        chunks.append(SemanticChunk(
                            chunkId=f"doc_{document_id}_p{page_num}_chk_{chunk_counter}",
                            documentId=document_id,
                            spaceId=space_id,
                            documentName=document_name,
                            pageNumber=page_num,
                            sectionPath=breadcrumb,
                            chunkType="TEXT",
                            content=chunk_text,
                            enrichedContent=enriched,
                            tokenCount=current_buffer_tokens
                        ))
                        chunk_counter += 1
                    current_buffer = []
                    current_buffer_tokens = 0

                # Thu thập toàn bộ khối mô tả hình ảnh
                diagram_lines = [line]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("#") and not lines[i].strip().startswith("[MÔ TẢ HÌNH ẢNH TRÊN TRANG"):
                    diagram_lines.append(lines[i])
                    i += 1

                diagram_text = "\n".join(diagram_lines).strip()
                breadcrumb = " > ".join(current_headers) if current_headers else (document_name or f"Tài liệu #{document_id}")
                enriched = f"[{breadcrumb} > Sơ đồ / Hình ảnh]\n{diagram_text}"
                chunks.append(SemanticChunk(
                    chunkId=f"doc_{document_id}_p{page_num}_diag_{chunk_counter}",
                    documentId=document_id,
                    spaceId=space_id,
                    documentName=document_name,
                    pageNumber=page_num,
                    sectionPath=breadcrumb,
                    chunkType="DIAGRAM",
                    content=diagram_text,
                    enrichedContent=enriched,
                    tokenCount=count_tokens_approx(diagram_text)
                ))
                chunk_counter += 1
                continue

            # 3. Phát hiện Bảng biểu Markdown
            if trimmed.startswith("|") and trimmed.endswith("|"):
                # Gom toàn bộ bảng
                table_lines = [line]
                i += 1
                while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                    table_lines.append(lines[i])
                    i += 1

                table_text = "\n".join(table_lines).strip()
                breadcrumb = " > ".join(current_headers) if current_headers else (document_name or f"Tài liệu #{document_id}")
                enriched = f"[{breadcrumb} > Bảng biểu]\n{table_text}"
                chunks.append(SemanticChunk(
                    chunkId=f"doc_{document_id}_p{page_num}_tbl_{chunk_counter}",
                    documentId=document_id,
                    spaceId=space_id,
                    documentName=document_name,
                    pageNumber=page_num,
                    sectionPath=breadcrumb,
                    chunkType="TABLE",
                    content=table_text,
                    enrichedContent=enriched,
                    tokenCount=count_tokens_approx(table_text)
                ))
                chunk_counter += 1
                continue

            # 4. Văn bản thông thường
            line_tokens = count_tokens_approx(trimmed)
            if current_buffer_tokens + line_tokens > max_tokens and current_buffer:
                chunk_text = "\n".join(current_buffer).strip()
                breadcrumb = " > ".join(current_headers) if current_headers else (document_name or f"Tài liệu #{document_id}")
                enriched = f"[{breadcrumb}]\n{chunk_text}"
                chunks.append(SemanticChunk(
                    chunkId=f"doc_{document_id}_p{page_num}_chk_{chunk_counter}",
                    documentId=document_id,
                    spaceId=space_id,
                    documentName=document_name,
                    pageNumber=page_num,
                    sectionPath=breadcrumb,
                    chunkType="TEXT",
                    content=chunk_text,
                    enrichedContent=enriched,
                    tokenCount=current_buffer_tokens
                ))
                chunk_counter += 1

                # Overlap giữ lại 1 vài dòng cuối
                overlap_buffer = []
                overlap_count = 0
                for prev_line in reversed(current_buffer):
                    prev_tokens = count_tokens_approx(prev_line)
                    if overlap_count + prev_tokens <= overlap_tokens:
                        overlap_buffer.insert(0, prev_line)
                        overlap_count += prev_tokens
                    else:
                        break
                current_buffer = overlap_buffer
                current_buffer_tokens = overlap_count

            current_buffer.append(line)
            current_buffer_tokens += line_tokens
            i += 1

        # Đóng gói phần còn lại của trang
        if current_buffer:
            chunk_text = "\n".join(current_buffer).strip()
            if chunk_text:
                breadcrumb = " > ".join(current_headers) if current_headers else (document_name or f"Tài liệu #{document_id}")
                enriched = f"[{breadcrumb}]\n{chunk_text}"
                chunks.append(SemanticChunk(
                    chunkId=f"doc_{document_id}_p{page_num}_chk_{chunk_counter}",
                    documentId=document_id,
                    spaceId=space_id,
                    documentName=document_name,
                    pageNumber=page_num,
                    sectionPath=breadcrumb,
                    chunkType="TEXT",
                    content=chunk_text,
                    enrichedContent=enriched,
                    tokenCount=current_buffer_tokens
                ))
                chunk_counter += 1

    logger.info(f"Hoàn thành phân đoạn tài liệu '{document_name}': Đã sinh ra tổng cộng {len(chunks)} Semantic Chunks.")
    return chunks
