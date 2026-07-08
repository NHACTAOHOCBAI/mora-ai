import json
from typing import List
from loguru import logger
from google import genai
from google.genai import types

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse, Citation

def generate_chat_response(request: ChatRequest) -> ChatResponse:
    logger.info(f"Bắt đầu xử lý câu hỏi: {request.question}")

    # Khởi tạo client Gemini
    client = genai.Client(api_key=settings.gemini_api_key)

    # 1. Định dạng ngữ cảnh (Context)
    context_str = ""
    for item in request.context:
        doc_name = item.documentName if item.documentName else f"Tài liệu #{item.documentId}"
        context_str += f"Tài liệu: {doc_name} (ID: {item.documentId}) - Trang {item.pageNumber}\nNội dung:\n{item.text}\n---\n"

    # 2. Định dạng lịch sử chat
    history_str = ""
    for h in request.history:
        role = "Người dùng" if h.sender == "user" else "Trợ lý AI"
        history_str += f"{role}: {h.text}\n"

    # 3. Xây dựng System Instructions và Prompt
    system_instruction = (
        "Bạn là Trợ lý Học tập AI tích hợp trong hệ thống Mora.\n"
        "Nhiệm vụ của bạn là trả lời các câu hỏi học thuật từ người dùng dựa trên ngữ cảnh tài liệu được cung cấp phía dưới.\n"
        "Hãy tuân thủ các quy tắc sau một cách nghiêm ngặt:\n"
        "1. Trả lời trung thực, khách quan và chính xác dựa trên tài liệu. Không bịa đặt hoặc suy diễn vượt quá tài liệu.\n"
        "2. Nếu tài liệu không có thông tin để trả lời câu hỏi, hãy trả lời rõ ràng rằng bạn không tìm thấy thông tin này trong tài liệu.\n"
        "3. Trích dẫn nguồn cụ thể cho các thông tin quan trọng. Mỗi trích dẫn (citation) cần có đúng số trang (pageNumber), đoạn trích nguyên văn (quote), và thông tin tài liệu (documentId, documentName) nếu có.\n"
        "4. Phản hồi bằng tiếng Việt trôi chảy, rõ ràng. Bạn có thể sử dụng bảng Markdown hoặc danh sách để so sánh/liệt kê thông tin nếu thấy phù hợp."
    )

    prompt = (
        f"--- NGỮ CẢNH TÀI LIỆU ---\n{context_str}\n"
        f"--- LỊCH SỬ HỘI THOẠI ---\n{history_str}\n"
        f"--- CÂU HỎI MỚI ---\nNgười dùng: {request.question}\n"
    )

    logger.info("Đang gọi Gemini API để sinh câu trả lời...")
    try:
        response = client.models.generate_content(
            model=settings.gemini_model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ChatResponse,
                temperature=settings.gemini_temperature
            )
        )

        # Parse kết quả JSON trả về từ Gemini
        result_json = response.text
        logger.info(f"Gemini API Response: {result_json}")
        
        parsed_data = json.loads(result_json)
        
        # Tạo danh sách Citation
        citations = []
        if "citations" in parsed_data:
            for cit in parsed_data["citations"]:
                # Tìm kiếm documentId và documentName thực tế từ context tương ứng với trang
                page_num = cit.get("pageNumber")
                doc_id = None
                doc_name = None
                
                # Ánh xạ ngược lại thông tin tài liệu từ context đầu vào
                for ctx in request.context:
                    if ctx.pageNumber == page_num:
                        doc_id = ctx.documentId
                        doc_name = ctx.documentName
                        break
                
                citations.append(Citation(
                    pageNumber=page_num,
                    quote=cit.get("quote", ""),
                    documentId=doc_id,
                    documentName=doc_name
                ))

        chat_response = ChatResponse(
            answer=parsed_data.get("answer", ""),
            citations=citations,
            condensedQuestion=request.question,
            promptSent=system_instruction + "\n\n" + prompt
        )
        return chat_response

    except Exception as e:
        logger.error(f"Lỗi khi gọi Gemini API: {e}", exc_info=True)
        return ChatResponse(
            answer="Đã xảy ra lỗi hệ thống khi AI đang xử lý câu hỏi của bạn. Vui lòng thử lại sau.",
            citations=[],
            condensedQuestion=request.question,
            promptSent=prompt
        )
