import json
from typing import List
from loguru import logger
from google import genai
from google.genai import types

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse, Citation, ChatSummarizeRequest, ChatSummarizeResponse

def generate_chat_response(request: ChatRequest) -> ChatResponse:
    logger.info(f"Bắt đầu xử lý câu hỏi: {request.question}")
    logger.info(f"Chat summary từ request: {request.chat_summary}")

    # Khởi tạo client Gemini
    client = genai.Client(api_key=settings.gemini_api_key)

    # 1. Định dạng ngữ cảnh (Context)
    context_str = ""
    for item in request.context:
        doc_name = item.documentName if item.documentName else f"Tài liệu #{item.documentId}"
        context_str += f"Tài liệu: {doc_name} (ID: {item.documentId}) - Trang {item.pageNumber}\nNội dung:\n{item.text}\n---\n"

    # 2. Phân tách Short-term Memory (Giới hạn tối đa 6 câu thoại thô gần nhất)
    # 6 tin nhắn tương đương khoảng 3 lượt hội thoại qua lại
    short_term_history = request.history[-6:] if len(request.history) > 6 else request.history

    history_str = ""
    for h in short_term_history:
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

    if request.chat_summary:
        system_instruction += (
            f"\n\n--- TÓM TẮT LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ (BỘ NHỚ DÀI HẠN) ---\n"
            f"{request.chat_summary}\n"
            f"Hãy sử dụng thông tin tóm tắt trên để hiểu ngữ cảnh dài hạn và các tham chiếu từ thay thế (như 'ông ấy', 'nó', 'dự án đó') nếu có."
        )

    prompt = (
        f"--- NGỮ CẢNH TÀI LIỆU ---\n{context_str}\n"
        f"--- LỊCH SỬ HỘI THOẠI (BỘ NHỚ NGẮN HẠN) ---\n{history_str}\n"
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

def generate_chat_summary(request: ChatSummarizeRequest) -> ChatSummarizeResponse:
    logger.info("Bắt đầu tóm tắt lịch sử hội thoại...")
    client = genai.Client(api_key=settings.gemini_api_key)

    # Định dạng lịch sử hội thoại thành chuỗi văn bản
    history_str = ""
    for h in request.history:
        role = "Người dùng" if h.sender == "user" else "Trợ lý AI"
        history_str += f"{role}: {h.text}\n"

    system_instruction = (
        "Nhiệm vụ của bạn là tóm tắt lịch sử hội thoại giữa Người dùng và Trợ lý AI một cách ngắn gọn, súc tích.\n"
        "Hãy tập trung vào các thông tin quan trọng: chủ đề thảo luận, câu hỏi cốt lõi của người dùng, và câu trả lời chính của trợ lý.\n"
        "Hãy viết bản tóm tắt bằng tiếng Việt dưới dạng một đoạn văn ngắn (không quá 150 từ)."
    )

    prompt = ""
    if request.previous_summary:
        new_messages = request.history[-2:] if len(request.history) >= 2 else request.history
        new_history_str = ""
        for h in new_messages:
            role = "Người dùng" if h.sender == "user" else "Trợ lý AI"
            new_history_str += f"{role}: {h.text}\n"

        prompt += (
            f"Bản tóm tắt lịch sử hội thoại trước đó:\n{request.previous_summary}\n\n"
            f"Các câu thoại mới nhất diễn ra:\n{new_history_str}\n\n"
            f"Nhiệm vụ của bạn là tích hợp các câu thoại mới nhất vào bản tóm tắt cũ và viết lại một bản tóm tắt mới hoàn chỉnh, ngắn gọn."
        )
    else:
        prompt += "Toàn bộ lịch sử hội thoại:\n"
        prompt += history_str

    try:
        response = client.models.generate_content(
            model=settings.gemini_model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            )
        )
        summary = response.text.strip()
        logger.info(f"Tóm tắt hội thoại thành công: {summary}")
        return ChatSummarizeResponse(summary=summary)
    except Exception as e:
        logger.error(f"Lỗi khi tóm tắt hội thoại bằng Gemini: {e}", exc_info=True)
        # Fallback trả về tóm tắt cũ hoặc tóm tắt mặc định
        return ChatSummarizeResponse(summary=request.previous_summary if request.previous_summary else "Hội thoại học tập về tài liệu.")
