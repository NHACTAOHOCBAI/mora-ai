import json
from fastapi import APIRouter, HTTPException
from app.schemas.study import StudyGenerateRequest, StudyGenerateResponse
from app.services.openai_service import get_client, generate, AI_ERROR, AI_CONFIGURATION_ERROR
from app.core.runtime_config import RuntimeConfigurationError
from app.services.vector_service import vector_store
from app.core.config import settings
from app.core.usage_tracking import collect_usage

router = APIRouter()

@router.post('/study/generate', response_model=StudyGenerateResponse)
def generate_study_set(request: StudyGenerateRequest):
    try:
        with collect_usage() as usage:
            chunks = vector_store.sample_chunks(request.spaceId, request.documentIds, per_document=8)
            if chunks:
                context_rows = [
                    {"documentId": c["documentId"], "documentName": c["documentName"], "pageNumber": c["pageNumber"], "text": c.get("content") or c.get("enrichedContent") or ""}
                    for c in chunks
                ]
            else:
            # The database pages remain a reliable source if Qdrant was reset or
            # indexing is temporarily unavailable. Keep the prompt bounded and
            # sample pages across each selected document.
                grouped = {doc_id: [] for doc_id in request.documentIds}
                for item in request.context:
                    if item.documentId in grouped and item.text.strip():
                        grouped[item.documentId].append(item)
                context_rows = []
                for doc_id in request.documentIds:
                    pages = grouped[doc_id]
                    step = max(1, len(pages) // 8)
                    context_rows.extend({"documentId": p.documentId, "documentName": p.documentName, "pageNumber": p.pageNumber, "text": p.text} for p in pages[::step][:8])
                if not context_rows:
                    raise HTTPException(status_code=422, detail='Tài liệu được chọn không có nội dung để tạo bộ ôn tập.')
            context = '\n---\n'.join(
                f"TÀI LIỆU id={c['documentId']}, tên={c['documentName']}, trang={c['pageNumber']}\n{c['text']}"
                for c in context_rows
            )
            instruction = """Bạn tạo nội dung ôn tập tiếng Việt chỉ dựa trên ngữ cảnh tài liệu.
Trả đúng JSON schema. Tạo chính xác số flashcard được yêu cầu và chính xác 5 câu trắc nghiệm.
Flashcard: mặt trước là câu hỏi/khái niệm, mặt sau trả lời rõ ràng. Quiz có đúng 4 lựa chọn, một đáp án đúng.
Không bịa thông tin. Mỗi mục phải có citation với documentId, documentName, pageNumber và quote ngắn xuất hiện nguyên văn trong ngữ cảnh.
Tiêu đề ngắn gọn, mô tả chủ đề chung của bộ tài liệu."""
            prompt = f"Số flashcard: {request.cardCount}\n\nNGỮ CẢNH:\n{context}"
            response = generate(get_client(), settings.openai_model, prompt, instruction, StudyGenerateResponse, operation="study_generation")
            result = StudyGenerateResponse.model_validate(json.loads(response))
            return {**result.model_dump(), "usage": usage}
    except HTTPException:
        raise
    except RuntimeConfigurationError:
        raise HTTPException(status_code=503, detail=AI_CONFIGURATION_ERROR) from None
    except Exception:
        raise HTTPException(status_code=503, detail=AI_ERROR) from None
