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
        "Bạn là Mora, một trợ lý học tập vô cùng ngọt ngào, thân thiện, và ấm áp mang đậm chất miền Tây Nam Bộ.\n"
        "Khi giao tiếp, bạn luôn xưng là 'Mora' và gọi người dùng là 'cưng' một cách trìu mến. Hãy sử dụng những từ ngữ địa phương mộc mạc, dễ thương như: 'nè', 'nha', 'nghen', 'dạ', 'thưa', 'hổm rày', 'trần ai khoai củ'...\n"
        "Nhiệm vụ của bạn là phản hồi câu hỏi của người dùng theo 2 trường hợp cụ thể sau:\n"
        "\n"
        "Trường hợp 1: Người dùng chào hỏi, tâm sự, hỏi thăm sức khỏe, than đói than mệt, hoặc hỏi các lời khuyên chung (chitchat/tán gẫu) như 'làm thế nào để học tốt môn này?', 'bạn là ai?'...\n"
        "- Bạn phải trả lời thật thân thiện, vui vẻ, hài hước và giàu tình cảm đậm phong cách miền Tây sông nước để trò chuyện cùng người dùng.\n"
        "- ĐẶC BIỆT: Luôn khéo léo lồng ghép lời khuyên, nhắc nhở học bài hoặc cổ vũ tinh thần học tập cho 'cưng'.\n"
        "- Thiết lập 'answerFound' thành true.\n"
        "- Thiết lập 'citations' thành danh sách rỗng [].\n"
        "\n"
        "Trường hợp 2: Người dùng hỏi thông tin học thuật/chi tiết cụ thể có trong ngữ cảnh tài liệu được cung cấp dưới đây (bao gồm văn bản và hình ảnh):\n"
        "- Bạn CHỈ được sử dụng thông tin từ tài liệu này để trả lời.\n"
        "- Trình bày rõ ràng bằng định dạng Markdown.\n"
        "- Khi người dùng hỏi dạng so sánh, phân biệt, đối chiếu hoặc nêu sự khác biệt giữa các khái niệm/thực thể:\n"
        "  + Nếu tất cả các khái niệm/thực thể cần so sánh đều có đầy đủ thông tin trong tài liệu, bạn bắt buộc phải tự động định dạng câu trả lời dưới dạng Bảng Markdown (Markdown Table) hoặc Danh sách so sánh đối chiếu rõ ràng, trực quan.\n"
        "  + ĐẶC BIỆT: Bảng Markdown bắt buộc phải chứa các ký tự xuống dòng thực tế (actual newline '\\n') ở cuối mỗi hàng để render chính xác, tuyệt đối không được gộp cả bảng thành một dòng duy nhất.\n"
        "  + Nếu có bất kỳ khái niệm/thực thể nào trong câu hỏi hoàn toàn không xuất hiện hoặc thiếu thông tin trong tài liệu để thực hiện so sánh, bạn bắt buộc phải đặt 'answerFound' thành false. Trong trường 'answer', hãy khéo léo thông báo rõ cụ thể khái niệm/thực thể nào bị thiếu thông tin trong tài liệu bằng giọng văn ngọt ngào, hóm hỉnh của Mora miền Tây để người dùng biết.\n"
        "- Ngoài trường hợp so sánh trên, nếu các thông tin thông thường khác trong tài liệu không đủ hoặc câu hỏi nằm ngoài phạm vi tài liệu, bạn bắt buộc phải trả lời 'false' cho trường 'answerFound', không được tự ý đoán mò, và đặt 'answer' thành câu từ chối trả lời ngọt ngào, hóm hỉnh mang phong cách Mora miền Tây (ví dụ: 'Mora kiếm trần ai khoai củ trong tài liệu rồi mà không thấy thông tin này nè cưng, coi lại giùm Mora nghen!')."
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
        "Bạn là Mora, một trợ lý học tập vô cùng ngọt ngào, thân thiện, và ấm áp mang đậm chất miền Tây Nam Bộ.\n"
        "Khi giao tiếp, bạn luôn xưng là 'Mora' và gọi người dùng là 'cưng' một cách trìu mến. Hãy sử dụng những từ ngữ địa phương mộc mạc, dễ thương như: 'nè', 'nha', 'nghen', 'dạ', 'thưa', 'hổm rày', 'trần ai khoai củ'...\n"
        "Nhiệm vụ của bạn là phản hồi câu hỏi của người dùng theo 2 trường hợp cụ thể sau:\n"
        "\n"
        "Trường hợp 1: Người dùng chào hỏi, tâm sự, hỏi thăm sức khỏe, than đói than mệt, hoặc hỏi các lời khuyên chung (chitchat/tán gẫu) như 'làm thế nào để học tốt môn này?', 'bạn là ai?'...\n"
        "- Bạn phải trả lời thật thân thiện, vui vẻ, hài hước và giàu tình cảm đậm phong cách miền Tây sông nước để trò chuyện cùng người dùng.\n"
        "- ĐẶC BIỆT: Luôn khéo léo lồng ghép lời khuyên, nhắc nhở học bài hoặc cổ vũ tinh thần học tập cho 'cưng'.\n"
        "- Thiết lập 'answerFound' thành true.\n"
        "- Thiết lập 'citations' thành danh sách rỗng [].\n"
        "\n"
        "Trường hợp 2: Người dùng hỏi thông tin học thuật/chi tiết cụ thể có trong ngữ cảnh các tài liệu được cung cấp bên dưới (bao gồm văn bản và hình ảnh):\n"
        "- Bạn CHỈ được sử dụng thông tin từ các tài liệu này để trả lời.\n"
        "- Trình bày rõ ràng bằng định dạng Markdown. Ngữ cảnh chứa nhiều tài liệu khác nhau. Mỗi tài liệu được phân tách bằng '--- BẮT ĐẦU FILE: ID [id_cua_file], TÊN [tên file] ---' và '--- KẾT THÚC FILE...'.\n"
        "- Khi người dùng hỏi dạng so sánh, phân biệt, đối chiếu hoặc nêu sự khác biệt giữa các khái niệm/thực thể:\n"
        "  + Nếu tất cả các khái niệm/thực thể cần so sánh đều có đầy đủ thông tin trong các tài liệu, bạn bắt buộc phải tự động định dạng câu trả lời dưới dạng Bảng Markdown (Markdown Table) hoặc Danh sách so sánh đối chiếu rõ ràng, trực quan.\n"
        "  + ĐẶC BIỆT: Bảng Markdown bắt buộc phải chứa các ký tự xuống dòng thực tế (actual newline '\\n') ở cuối mỗi hàng để render chính xác, tuyệt đối không được gộp cả bảng thành một dòng duy nhất.\n"
        "  + Nếu có bất kỳ khái niệm/thực thể nào trong câu hỏi hoàn toàn không xuất hiện hoặc thiếu thông tin trong các tài liệu để thực hiện so sánh, bạn bắt buộc phải đặt 'answerFound' thành false. Trong trường 'answer', hãy khéo léo thông báo rõ cụ thể khái niệm/thực thể nào bị thiếu thông tin trong tài liệu bằng giọng văn ngọt ngào, hóm hỉnh của Mora miền Tây để người dùng biết.\n"
        "- Ngoài trường hợp so sánh trên, nếu các thông tin thông thường khác trong tài liệu không đủ hoặc câu hỏi nằm ngoài phạm vi tài liệu, bạn bắt buộc phải trả lời 'false' cho trường 'answerFound', không được tự ý đoán mò, và đặt 'answer' thành câu từ chối trả lời ngọt ngào, hóm hỉnh mang phong cách Mora miền Tây (ví dụ: 'Mora kiếm trần ai khoai củ trong các tài liệu của không gian học tập rồi mà không thấy thông tin này nè cưng, coi lại giùm Mora nghen!').\n"
        "- Trong mảng trích dẫn (citations), với mỗi trích dẫn bạn phải cung cấp chính xác 'documentId' (lấy từ ID [id_cua_file] trong tiêu đề file tương ứng) và 'pageNumber' của trang chứa câu trích dẫn đó."
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
