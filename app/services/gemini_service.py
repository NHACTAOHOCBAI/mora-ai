import base64
import json
from typing import List, Optional
from google import genai
from google.genai import types
from app.core.config import settings
from app.schemas.chat import (
    ChatMessageDto,
    DocumentChatResponse,
    SpaceChatResponse,
    StudyNotesResponse
)

# Khởi tạo Gemini Client
client = genai.Client(api_key=settings.gemini_api_key)

def condense_question(history: List[ChatMessageDto], question: str) -> str:
    if not history:
        return question
    
    history_str = ""
    for msg in history:
        role = "User" if msg.sender.lower() == "user" else "Assistant"
        history_str += f"{role}: {msg.text}\n"
        
    system_instruction = (
        "Bạn là một trợ lý ngôn ngữ AI thông minh.\n"
        "Nhiệm vụ của bạn là kết hợp lịch sử cuộc trò chuyện gần nhất và câu hỏi mới của người dùng thành một \"Câu hỏi độc lập\" (Standalone Question) hoàn chỉnh, rõ nghĩa, và tự chứa đầy đủ ngữ cảnh để có thể dùng truy vấn trực tiếp vào tài liệu.\n"
        "- Không được trả lời câu hỏi, CHỈ được viết lại câu hỏi.\n"
        "- Giữ nguyên ngôn ngữ của câu hỏi gốc (nếu là Tiếng Việt thì viết lại bằng Tiếng Việt).\n"
        "- Nếu câu hỏi mới đã đầy đủ nghĩa và không phụ thuộc vào lịch sử chat, hãy trả về chính xác câu hỏi mới đó."
    )
    
    prompt_text = f"Lịch sử trò chuyện:\n{history_str}\n\nCâu hỏi mới: {question}\n\nHãy viết lại câu hỏi độc lập:"
    try:
        response = client.models.generate_content(
            model=settings.gemini_model_name,
            contents=[prompt_text],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=settings.gemini_temperature,
            )
        )
        rewritten = response.text.strip() if response.text else question
        return rewritten if rewritten else question
    except Exception as e:
        print(f"Failed to condense question: {e}")
        return question

def chat_with_document_service(
    context: str,
    question: str,
    base64_images: List[str],
    history: List[ChatMessageDto]
) -> DocumentChatResponse:
    condensed_q = condense_question(history, question)
    
    system_instruction = (
        "Bạn là một trợ lý học thuật nghiêm khắc.\n"
        "Hãy trả lời câu hỏi của người dùng CHỈ sử dụng thông tin từ ngữ cảnh tài liệu được cung cấp dưới đây (bao gồm cả nội dung văn bản và hình ảnh của tài liệu đó).\n"
        "Hãy trình bày câu trả lời một cách trực quan, có bố cục rõ ràng bằng cách sử dụng định dạng Markdown (ví dụ: sử dụng danh sách gạch đầu dòng, danh sách số, bôi đậm các thuật ngữ hoặc ý quan trọng, chia đoạn mạch lạc) để người dùng dễ đọc.\n"
        "Nếu tài liệu có hình ảnh đính kèm (hoặc bản thân tài liệu là hình ảnh), hãy phân tích kỹ hình ảnh và bạn ĐƯỢC PHÉP suy luận logic dựa trên hình ảnh để trả lời câu hỏi của người dùng.\n"
        "Nếu thông tin trong tài liệu (cả phần chữ và phần hình ảnh) không đủ hoặc câu hỏi nằm ngoài phạm vi tài liệu, bạn bắt buộc phải trả lời 'false' cho trường 'answerFound', không được tự ý đoán mò, và đặt 'answer' thành câu từ chối trả lời phù hợp (Ví dụ: \"Tôi không tìm thấy thông tin này trong tài liệu.\")."
    )
    
    prompt_text = f"Ngữ cảnh tài liệu:\n{context}\n\nCâu hỏi: {condensed_q}"
    contents = [prompt_text]
    
    for b64 in base64_images:
        try:
            image_data = base64.b64decode(b64)
            contents.append(
                types.Part.from_bytes(
                    data=image_data,
                    mime_type="image/jpeg"
                )
            )
        except Exception as e:
            raise ValueError(f"Invalid base64 image data in list: {str(e)}")

    full_prompt = (
        f"[SYSTEM PROMPT]\n{system_instruction}\n\n"
        f"[USER MESSAGE]\nNgữ cảnh tài liệu:\n{context}\n\nCâu hỏi: {condensed_q}"
    )

    response = client.models.generate_content(
        model=settings.gemini_model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
            response_schema=DocumentChatResponse,
        )
    )
    
    res_data = json.loads(response.text)
    res_obj = DocumentChatResponse(**res_data)
    res_obj.condensedQuestion = condensed_q
    res_obj.promptSent = full_prompt
    return res_obj

def chat_with_space_service(
    context: str,
    question: str,
    base64_images: List[str],
    history: List[ChatMessageDto]
) -> SpaceChatResponse:
    condensed_q = condense_question(history, question)

    system_instruction = (
        "Bạn là một trợ lý học thuật nghiêm khắc.\n"
        "Hãy trả lời câu hỏi của người dùng CHỈ sử dụng thông tin từ ngữ cảnh tài liệu được cung cấp dưới đây (bao gồm cả nội dung văn bản và hình ảnh của tài liệu đó).\n"
        "Hãy trình bày câu trả lời một cách trực quan, có bố cục rõ ràng bằng cách sử dụng định dạng Markdown (ví dụ: sử dụng danh sách gạch đầu dòng, danh sách số, bôi đậm các thuật ngữ hoặc ý quan trọng, chia đoạn mạch lạc) để người dùng dễ đọc.\n"
        "Ngữ cảnh chứa nhiều tài liệu khác nhau. Mỗi tài liệu được phân tách bằng '--- BẮT ĐẦU FILE: ID [id_cua_file], TÊN [tên file] ---' và '--- KẾT THÚC FILE...'.\n"
        "Nếu tài liệu có hình ảnh đính kèm (hoặc bản thân tài liệu là hình ảnh), hãy phân tích kỹ hình ảnh và bạn ĐƯỢC PHÉP suy luận logic dựa trên hình ảnh để trả lời câu hỏi của người dùng.\n"
        "Nếu thông tin trong các tài liệu (cả phần chữ và phần hình ảnh) không đủ hoặc câu hỏi nằm ngoài phạm vi tài liệu, bạn bắt buộc phải trả lời 'false' cho trường 'answerFound', không được tự ý đoán mò, và đặt 'answer' thành câu từ chối trả lời phù hợp (Ví dụ: \"Tôi không tìm thấy thông tin này trong các tài liệu của không gian học tập.\").\n"
        "Trong mảng trích dẫn (citations), với mỗi trích dẫn bạn phải cung cấp chính xác 'documentId' (lấy từ ID [id_cua_file] trong tiêu đề file tương ứng) và 'pageNumber' của trang chứa câu trích dẫn đó."
    )
    
    prompt_text = f"Ngữ cảnh tài liệu:\n{context}\n\nCâu hỏi: {condensed_q}"
    contents = [prompt_text]
    
    for b64 in base64_images:
        try:
            image_data = base64.b64decode(b64)
            contents.append(
                types.Part.from_bytes(
                    data=image_data,
                    mime_type="image/jpeg"
                )
            )
        except Exception as e:
            raise ValueError(f"Invalid base64 image data in list: {str(e)}")

    full_prompt = (
        f"[SYSTEM PROMPT]\n{system_instruction}\n\n"
        f"[USER MESSAGE]\nNgữ cảnh tài liệu:\n{context}\n\nCâu hỏi: {condensed_q}"
    )

    response = client.models.generate_content(
        model=settings.gemini_model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
            response_schema=SpaceChatResponse,
        )
    )
    
    res_data = json.loads(response.text)
    res_obj = SpaceChatResponse(**res_data)
    res_obj.condensedQuestion = condensed_q
    res_obj.promptSent = full_prompt
    return res_obj

def generate_study_notes_service(context: str) -> StudyNotesResponse:
    system_instruction = (
        "Bạn là một chuyên gia tóm tắt tài liệu và giảng dạy học thuật.\n"
        "Nhiệm vụ của bạn là đọc nội dung tài liệu được cung cấp và sinh ra 2 phần:\n"
        "1. Tóm tắt nội dung tài liệu (dưới dạng Markdown chi tiết, cấu trúc rõ ràng, sinh động, dễ học).\n"
        "2. Một danh sách gồm khoảng 5-10 câu hỏi ôn tập (Flashcards) dưới dạng định dạng JSON chuẩn. Mỗi flashcard có cấu trúc: {\"question\": \"câu hỏi...\", \"answer\": \"câu trả lời...\"}\n"
        "Hãy đảm bảo bạn chỉ sử dụng thông tin trong tài liệu đã cung cấp."
    )
    
    prompt_text = f"Tài liệu:\n{context}"
    
    response = client.models.generate_content(
        model=settings.gemini_model_name,
        contents=[prompt_text],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
            response_schema=StudyNotesResponse,
        )
    )
    
    res_data = json.loads(response.text)
    return StudyNotesResponse(**res_data)
