import json
from typing import List, Tuple, Optional
from pydantic import BaseModel
from loguru import logger
from google import genai
from google.genai import types

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse, Citation, ChatSummarizeRequest, ChatSummarizeResponse

# Definitions for structured JSON schema response from agents
class CitationSchema(BaseModel):
    pageNumber: int
    quote: str
    documentId: int | None = None
    documentName: str | None = None

class RAGResponseSchema(BaseModel):
    answer: str
    citations: List[CitationSchema]
    condensedQuestion: str

class RouteSchema(BaseModel):
    intent: str  # "RAG" or "GENERAL"
    reason: str

class EvaluationSchema(BaseModel):
    is_faithful: bool
    hallucinations: List[str]
    score: float  # 0.0 to 1.0

class MultiAgentOrchestrator:
    def __init__(self, api_key: str, chat_model: str = None, router_model: str = None, evaluator_model: str = None):
        if not api_key:
            raise ValueError("Gemini API Key is required.")
        self.client = genai.Client(api_key=api_key)
        self.chat_model = chat_model or settings.gemini_model_name
        self.router_model = router_model or settings.gemini_model_name
        self.evaluator_model = evaluator_model or settings.gemini_evaluator_model_name

    def route_agent(self, question: str, chat_summary: str, history: List[dict]) -> str:
        logger.info(f"[Router Agent] Classifying intent using model '{self.router_model}' for: '{question}'")
        history_str = "\n".join([f"{h.get('sender')}: {h.get('text')}" for h in history[-4:]])
        system_instruction = (
            "Bạn là trợ lý định tuyến (routing agent) cho hệ thống Multi-Agent.\n"
            "Nhiệm vụ của bạn là phân loại xem câu hỏi của người dùng có yêu cầu thông tin từ tài liệu đã tải lên của họ (sách giáo trình, bài giảng PDF, ghi chú học tập) hay đó là một cuộc trò chuyện/yêu cầu chung.\n\n"
            "Quy tắc:\n"
            "1. Phân loại là 'RAG' nếu câu hỏi đề cập đến tài liệu học tập, các slide cụ thể, nội dung bài học, công thức trong tài liệu hoặc các thuật ngữ chuyên sâu liên quan đến môn học.\n"
            "2. Phân loại là 'GENERAL' nếu đó là cuộc trò chuyện thông thường (chitchat), yêu cầu viết code, viết email, giải toán chung, lịch sử chung, dịch thuật hoặc khi câu hỏi rõ ràng không cần ngữ cảnh tài liệu.\n"
            "3. Trả về phản hồi theo đúng cấu trúc JSON được yêu cầu."
        )
        prompt = (
            f"--- Lịch sử cuộc trò chuyện ---\n{history_str}\n"
            f"--- Tóm tắt lịch sử trước đó ---\n{chat_summary}\n"
            f"--- Câu hỏi mới ---\n{question}\n"
        )
        try:
            response = self.client.models.generate_content(
                model=self.router_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=RouteSchema,
                    temperature=0.0
                )
            )
            result = json.loads(response.text)
            intent = result.get("intent", "GENERAL").upper()
            logger.info(f"[Router Agent] Decision: {intent} (Reason: {result.get('reason')})")
            return intent
        except Exception as e:
            logger.error(f"[Router Agent] Error during routing with model {self.router_model}: {e}", exc_info=True)
            return "GENERAL"

    def retrieval_agent(self, question: str, space_id: Optional[int] = None, raw_context: List[dict] = None) -> List[dict]:
        logger.info(f"[Retrieval Agent] Bắt đầu xử lý truy vấn cho Space ID: {space_id}")
        if space_id:
            try:
                from app.services.retrieval_service import hybrid_retrieve
                retrieved_chunks = hybrid_retrieve(
                    query=question,
                    space_id=space_id,
                    gemini_client=self.client
                )
                if retrieved_chunks:
                    mapped_context = []
                    for c in retrieved_chunks:
                        content_text = c.get("enrichedContent") or c.get("content") or ""
                        mapped_context.append({
                            "pageNumber": c.get("pageNumber", 1),
                            "text": content_text,
                            "documentName": c.get("documentName", ""),
                            "documentId": c.get("documentId"),
                            "sectionPath": c.get("sectionPath", "")
                        })
                    logger.info(f"[Retrieval Agent] Hybrid Search thành công! Chọn {len(mapped_context)} chunks tinh hoa cung cấp cho Synthesis Agent.")
                    return mapped_context
            except Exception as e:
                logger.error(f"[Retrieval Agent] Lỗi trong quá trình Hybrid Search: {e}", exc_info=True)

        logger.info(f"[Retrieval Agent] Sử dụng raw_context đầu vào. Tổng số: {len(raw_context) if raw_context else 0}")
        return raw_context or []

    def general_chat_agent(self, question: str, chat_summary: str, history: List[dict]) -> Tuple[str, str]:
        logger.info(f"[General Chat Agent] Answering query using model '{self.chat_model}': '{question}'")
        system_instruction = (
            "Bạn là Trợ lý Học tập AI tích hợp trong hệ thống Mora.\n"
            "Nhiệm vụ của bạn là trợ giúp người dùng giải quyết các câu hỏi học thuật chung (như giải thích lý thuyết, viết code, giải toán phổ thông, dịch thuật...).\n\n"
            "QUY TẮC ĐỊNH DẠNG & TRÌNH BÀY (RẤT QUAN TRỌNG):\n"
            "1. TUYỆT ĐỐI KHÔNG VIẾT DỒN CẢ CÂU TRẢ LỜI THÀNH MỘT ĐOẠN VĂN DÀI.\n"
            "2. Phân chia bố cục rõ ràng với các tiêu đề mục (###, ####), cách nhau bằng dòng trống (\\n\\n).\n"
            "3. Sử dụng danh sách gạch đầu dòng (- ) hoặc đánh số thứ tự (1., 2.) cho từng ý, mỗi ý nằm trên một dòng riêng biệt.\n"
            "4. In đậm (**từ khóa**, **khái niệm chính**) và sử dụng `inline code` cho thuật ngữ kỹ thuật, biến, hàm.\n"
            "5. Sử dụng khối code có highlight cú pháp hoặc Bảng Markdown (| Cột 1 | Cột 2 |) khi thích hợp.\n"
            "6. Nếu người dùng yêu cầu tạo đề kiểm tra, bài thi hoặc các câu hỏi trắc nghiệm/tự luận, hãy lịch sự từ chối và nhắc họ rằng bạn chỉ tập trung hỗ trợ giải đáp thắc mắc kiến thức.\n"
            "7. Nếu người dùng đề cập đến tài liệu học tập của họ, hãy lịch sự nhắc họ rằng đây là chế độ chat tự do và bạn không sử dụng tài liệu học tập cho câu hỏi này."
        )
        if chat_summary:
            system_instruction += f"\nTóm tắt lịch sử hội thoại trước đó: {chat_summary}"
        history_str = "\n".join([f"{h.get('sender')}: {h.get('text')}" for h in history[-6:]])
        prompt = (
            f"--- LỊCH SỬ HỘI THOẠI ---\n{history_str}\n"
            f"--- CÂU HỎI MỚI ---\nNgười dùng: {question}\n"
        )
        try:
            response = self.client.models.generate_content(
                model=self.chat_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7
                )
            )
            return response.text, system_instruction + "\n\n" + prompt
        except Exception as e:
            logger.error(f"[General Chat Agent] Error: {e}", exc_info=True)
            return "Đã xảy ra lỗi khi tạo phản hồi. Vui lòng kiểm tra lại API Key hoặc hạn mức Model.", prompt

    def synthesis_agent(self, question: str, context: List[dict], chat_summary: str, history: List[dict]) -> dict:
        logger.info(f"[Synthesis Agent] Synthesizing answer using model '{self.chat_model}' with {len(context)} chunks.")
        context_str = ""
        for item in context:
            doc_name = item.get("documentName", f"Tài liệu #{item.get('documentId')}")
            context_str += f"Tài liệu: {doc_name} (ID: {item.get('documentId')}) - Trang {item.get('pageNumber')}\nNội dung:\n{item.get('text')}\n---\n"
        history_str = "\n".join([f"{h.get('sender')}: {h.get('text')}" for h in history[-6:]])
        system_instruction = (
            "Bạn là Trợ lý Học tập AI tích hợp trong hệ thống Mora (Source-Grounded AI Learning Assistant).\n"
            "Nhiệm vụ của bạn là trả lời các câu hỏi học thuật từ người dùng dựa trên ngữ cảnh tài liệu được cung cấp phía dưới.\n\n"
            "HÃY TUÂN THỦ CÁC QUY TẮC SAU MỘT CÁCH NGHIÊM NGẶT:\n"
            "1. TÍNH TRUNG THỰC & CHÍNH XÁC: Trả lời trung thực, khách quan và chính xác dựa trên tài liệu. Không bịa đặt hoặc suy diễn vượt quá tài liệu.\n"
            "2. THIẾU THÔNG TIN: Nếu tài liệu không có thông tin để trả lời câu hỏi, hãy trả lời rõ ràng rằng bạn không tìm thấy thông tin này trong tài liệu.\n"
            "3. TRÍCH DẪN NGUỒN (CITATIONS): Trích dẫn nguồn cụ thể cho các thông tin quan trọng. Mỗi trích dẫn (citation) cần có đúng số trang (pageNumber), đoạn trích nguyên văn (quote), và thông tin tài liệu (documentId, documentName) nếu có.\n"
            "4. QUY CHUẨN ĐỊNH DẠNG MARKDOWN TRỰC QUAN, DỄ ĐỌC (BẮT BUỘC TUÂN THỦ):\n"
            "   - TUYỆT ĐỐI KHÔNG VIẾT DỒN CẢ CÂU TRẢ LỜI THÀNH MỘT ĐOẠN VĂN DÀI LIỀN TÙ TÌ.\n"
            "   - Chia câu trả lời thành các phần rõ ràng, phân cách bằng dòng trống (\\n\\n) giữa các đoạn.\n"
            "   - Sử dụng Tiêu đề Markdown (### hoặc ####) cho từng phần / đề mục chính.\n"
            "   - Sử dụng Danh sách gạch đầu dòng (- hoặc *) hoặc Danh sách đánh số (1., 2., 3.) cho các ý phân tích, mỗi ý nằm trên một dòng riêng biệt.\n"
            "   - In đậm (**từ khóa quan trọng**, **khái niệm cốt lõi**) để làm nổi bật kiến thức.\n"
            "   - Sử dụng inline code (`tên_thành_phần`, `biến`, `lệnh`, `thanh ghi`) khi nhắc đến các yếu tố kỹ thuật (ví dụ `processor0`, `cache`, `registers`).\n"
            "   - Khi mô tả sơ đồ / kiến trúc: Phân tách rõ ràng các cấp độ/tầng (layers), các khối thành phần và luồng tương tác giữa chúng bằng danh sách phân cấp hoặc bảng.\n"
            "   - Khi so sánh / đối chiếu: Bắt buộc sử dụng Bảng Markdown (| Cột 1 | Cột 2 |) để trình bày trực quan.\n"
            "5. TỪ CHỐI TẠO BÀI THI: Nếu người dùng yêu cầu tạo đề kiểm tra, bài thi hoặc các câu hỏi trắc nghiệm/tự luận ôn tập từ tài liệu, hãy lịch sự từ chối và nhắc họ rằng bạn chỉ tập trung hỗ trợ giải đáp thắc mắc kiến thức dựa trên nội dung tài liệu."
        )
        if chat_summary:
            system_instruction += f"\n\n--- TÓM TẮT LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ ---\n{chat_summary}"
        prompt = (
            f"--- NGỮ CẢNH TÀI LIỆU ---\n{context_str}\n"
            f"--- LỊCH SỬ HỘI THOẠI ---\n{history_str}\n"
            f"--- CÂU HỎI MỚI ---\nNgười dùng: {question}\n"
        )
        try:
            response = self.client.models.generate_content(
                model=self.chat_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=RAGResponseSchema,
                    temperature=0.0
                )
            )
            result = json.loads(response.text)
            result["promptSent"] = system_instruction + "\n\n" + prompt
            return result
        except Exception as e:
            logger.error(f"[Synthesis Agent] Error with model {self.chat_model}: {e}", exc_info=True)
            return {
                "answer": "Không thể tổng hợp câu trả lời dựa trên tài liệu. Vui lòng kiểm tra lại API Key hoặc hạn mức Model.",
                "citations": [],
                "condensedQuestion": question,
                "promptSent": prompt
            }

    def evaluator_agent(self, answer: str, context: List[dict]) -> Tuple[bool, float]:
        logger.info(f"[Evaluator Agent] Performing QC using model '{self.evaluator_model}'.")
        if not context:
            return True, 1.0
        context_str = "\n---\n".join([c.get("text", "") for c in context])
        system_instruction = (
            "Bạn là trợ lý kiểm định chất lượng (QC evaluator agent) trong hệ thống RAG.\n"
            "Nhiệm vụ của bạn là đánh giá xem câu trả lời được sinh ra có trung thực, chính xác dựa trên ngữ cảnh được cung cấp hay không và đảm bảo KHÔNG có lỗi hallucination (thông tin tự bịa).\n"
            "Quy tắc:\n"
            "1. Đọc kỹ Ngữ cảnh (Context) và Câu trả lời được sinh ra (Generated Answer).\n"
            "2. Phát hiện xem có phát biểu nào trong câu trả lời không được hỗ trợ bởi Ngữ cảnh hoặc mâu thuẫn với Ngữ cảnh hay không.\n"
            "3. Trả về giá trị boolean 'is_faithful' và điểm số 'score' từ 0.0 đến 1.0 (trong đó 1.0 là hoàn toàn trung thực/khớp với ngữ cảnh và 0.0 là hoàn toàn tự bịa).\n"
            "4. Trả về phản hồi theo đúng cấu trúc JSON được yêu cầu."
        )
        prompt = (
            f"--- Ngữ cảnh tài liệu ---\n{context_str}\n"
            f"--- Câu trả lời được sinh ra ---\n{answer}\n"
        )
        try:
            response = self.client.models.generate_content(
                model=self.evaluator_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=EvaluationSchema,
                    temperature=0.0
                )
            )
            result = json.loads(response.text)
            is_faithful = result.get("is_faithful", True)
            score = result.get("score", 1.0)
            logger.info(f"[Evaluator Agent] Quality score: {score} (Is Faithful: {is_faithful})")
            return is_faithful, score
        except Exception as e:
            logger.error(f"[Evaluator Agent] Evaluation error with model {self.evaluator_model}: {e}", exc_info=True)
            return True, 1.0


def generate_chat_response(request: ChatRequest) -> ChatResponse:
    if not request.api_key:
        logger.error("Yêu cầu chat bị từ chối do thiếu Gemini API Key.")
        raise ValueError("Vui lòng cấu hình Gemini API Key trước khi sử dụng.")

    logger.info(f"Bắt đầu xử lý câu hỏi: {request.question}")
    
    orchestrator = MultiAgentOrchestrator(
        api_key=request.api_key,
        chat_model=request.chat_model,
        router_model=request.router_model,
        evaluator_model=request.evaluator_model
    )
    
    # Chuẩn bị dữ liệu cho các Agent
    space_id = request.space_id or request.spaceId
    raw_context = [
        {
            "pageNumber": ctx.pageNumber,
            "text": ctx.text,
            "documentName": ctx.documentName,
            "documentId": ctx.documentId,
            "sectionPath": ctx.sectionPath
        } for ctx in (request.context or [])
    ]
    history = [
        {
            "sender": h.sender,
            "text": h.text
        } for h in request.history
    ]
    
    # 1. Router Agent quyết định hướng đi
    intent = orchestrator.route_agent(request.question, request.chat_summary or "", history)
    
    answer = ""
    citations = []
    condensed_question = request.question
    prompt_sent = ""
    
    # 2. Xử lý theo phân loại
    if intent == "RAG":
        filtered_context = orchestrator.retrieval_agent(
            question=request.question, 
            space_id=space_id, 
            raw_context=raw_context
        )
        if not filtered_context:
            logger.warn(f"[Orchestrator] Không tìm thấy ngữ cảnh nào cho câu hỏi RAG: '{request.question}'")
            answer = "Không tìm thấy thông tin phù hợp trong tài liệu của Không gian học tập này để trả lời câu hỏi của bạn."
            citations = []
            condensed_question = request.question
            prompt_sent = ""
        else:
            max_retries = 2
            for attempt in range(max_retries):
                logger.info(f"[Orchestrator] Synthesis attempt {attempt + 1}")
                rag_result = orchestrator.synthesis_agent(request.question, filtered_context, request.chat_summary or "", history)
                answer = rag_result.get("answer", "")
                citations = rag_result.get("citations", [])
                condensed_question = rag_result.get("condensedQuestion", request.question)
                prompt_sent = rag_result.get("promptSent", "")
                
                # Evaluator Agent kiểm QC câu trả lời
                is_faithful, score = orchestrator.evaluator_agent(answer, filtered_context)
                if is_faithful or score >= 0.7:
                    logger.info("[Orchestrator] QC passed successfully!")
                    break
                else:
                    logger.warning(f"[Orchestrator] QC failed with score {score}. Retrying synthesis...")
    else:
        answer, prompt_sent = orchestrator.general_chat_agent(request.question, request.chat_summary or "", history)
        citations = []
        condensed_question = request.question
        
    # Map và deduplicate citations sang DTO Citation (gom các quote cùng trang)
    unique_citations = {}
    for c in citations:
        page_num = c.get("pageNumber")
        if page_num is None:
            continue
        doc_id = c.get("documentId")
        doc_name = c.get("documentName")
        quote = (c.get("quote") or "").strip()
        key = (doc_id, doc_name, page_num)
        
        if key not in unique_citations:
            unique_citations[key] = {
                "pageNumber": page_num,
                "documentId": doc_id,
                "documentName": doc_name,
                "quotes": [quote] if quote else []
            }
        else:
            if quote and quote not in unique_citations[key]["quotes"]:
                unique_citations[key]["quotes"].append(quote)

    citations_mapped = [
        Citation(
            pageNumber=item["pageNumber"],
            quote="\n---\n".join(item["quotes"]),
            documentId=item["documentId"],
            documentName=item["documentName"]
        )
        for item in unique_citations.values()
    ]
        
    return ChatResponse(
        answer=answer,
        citations=citations_mapped,
        condensedQuestion=condensed_question,
        promptSent=prompt_sent
    )

def generate_chat_summary(request: ChatSummarizeRequest) -> ChatSummarizeResponse:
    if not request.api_key:
        logger.warning("Bỏ qua tóm tắt hội thoại do thiếu API Key.")
        return ChatSummarizeResponse(summary=request.previous_summary if request.previous_summary else "")

    logger.info("Bắt đầu tóm tắt lịch sử hội thoại...")
    client = genai.Client(api_key=request.api_key)
    model_name = request.summarizer_model or settings.gemini_model_name

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
        # Lấy tối đa 12 tin nhắn gần nhất (tương đương 6 lượt Q&A trong chu kỳ batching)
        new_messages = request.history[-12:] if len(request.history) >= 12 else request.history
        new_history_str = ""
        for h in new_messages:
            role = "Người dùng" if h.sender == "user" else "Trợ lý AI"
            new_history_str += f"{role}: {h.text}\n"

        prompt += (
            f"Bản tóm tắt lịch sử hội thoại trước đó:\n{request.previous_summary}\n\n"
            f"Các câu thoại mới nhất diễn ra trong chu kỳ vừa qua:\n{new_history_str}\n\n"
            f"Nhiệm vụ của bạn là tích hợp các nội dung mới vào bản tóm tắt cũ và viết lại một bản tóm tắt mới hoàn chỉnh, ngắn gọn (không quá 150 từ)."
        )
    else:
        prompt += "Toàn bộ lịch sử hội thoại:\n"
        prompt += history_str

    try:
        response = client.models.generate_content(
            model=model_name,
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
        return ChatSummarizeResponse(summary=request.previous_summary if request.previous_summary else "Hội thoại học tập về tài liệu.")

def validate_gemini_key(api_key: str, model_name: str = "gemini-2.5-flash") -> Tuple[bool, str]:
    if not api_key or not api_key.strip():
        return False, "API Key không được để trống."
    try:
        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model=model_name or "gemini-2.5-flash",
            contents="ping",
            config=types.GenerateContentConfig(
                max_output_tokens=5,
                temperature=0.0
            )
        )
        return True, "API Key hợp lệ và kết nối Google AI Studio thành công."
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Key validation failed: {error_msg}")
        return False, f"API Key không hợp lệ hoặc đã hết hạn mức: {error_msg}"

