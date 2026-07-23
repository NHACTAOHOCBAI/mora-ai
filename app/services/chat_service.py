import json
from typing import List, Tuple
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
    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def route_agent(self, question: str, chat_summary: str, history: List[dict]) -> str:
        logger.info(f"[Router Agent] Classifying intent for: '{question}'")
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
                model=settings.gemini_model_name,
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
            logger.error(f"[Router Agent] Error during routing: {e}", exc_info=True)
            return "GENERAL"

    def retrieval_agent(self, question: str, raw_context: List[dict]) -> List[dict]:
        logger.info(f"[Retrieval Agent] Filtering context chunks. Total input: {len(raw_context)}")
        return raw_context

    def general_chat_agent(self, question: str, chat_summary: str, history: List[dict]) -> Tuple[str, str]:
        logger.info(f"[General Chat Agent] Answering general query: '{question}'")
        system_instruction = (
            "Bạn là Trợ lý Học tập AI tích hợp trong hệ thống Mora.\n"
            "Nhiệm vụ của bạn là trợ giúp người dùng giải quyết các câu hỏi học thuật chung (như giải thích lý thuyết, viết code, giải toán phổ thông, dịch thuật...).\n"
            "Hãy trả lời một cách tự nhiên, chi tiết, chuyên nghiệp bằng tiếng Việt và sử dụng định dạng Markdown nếu cần thiết.\n"
            "Nếu người dùng yêu cầu tạo đề kiểm tra, bài thi hoặc các câu hỏi trắc nghiệm/tự luận, hãy lịch sự từ chối và nhắc họ rằng bạn chỉ tập trung hỗ trợ giải đáp thắc mắc kiến thức.\n"
            "Nếu người dùng đề cập đến tài liệu học tập của họ, hãy lịch sự nhắc họ rằng đây là chế độ chat tự do và bạn không sử dụng tài liệu học tập cho câu hỏi này."
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
                model=settings.gemini_model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7
                )
            )
            return response.text, system_instruction + "\n\n" + prompt
        except Exception as e:
            logger.error(f"[General Chat Agent] Error: {e}", exc_info=True)
            return "Đã xảy ra lỗi khi tạo phản hồi. Vui lòng thử lại sau.", prompt

    def synthesis_agent(self, question: str, context: List[dict], chat_summary: str, history: List[dict]) -> dict:
        logger.info(f"[Synthesis Agent] Synthesizing answer with {len(context)} context chunks.")
        context_str = ""
        for item in context:
            doc_name = item.get("documentName", f"Tài liệu #{item.get('documentId')}")
            context_str += f"Tài liệu: {doc_name} (ID: {item.get('documentId')}) - Trang {item.get('pageNumber')}\nNội dung:\n{item.get('text')}\n---\n"
        history_str = "\n".join([f"{h.get('sender')}: {h.get('text')}" for h in history[-6:]])
        system_instruction = (
            "Bạn là Trợ lý Học tập AI tích hợp trong hệ thống Mora.\n"
            "Nhiệm vụ của bạn là trả lời các câu hỏi học thuật từ người dùng dựa trên ngữ cảnh tài liệu được cung cấp phía dưới.\n"
            "Hãy tuân thủ các quy tắc sau một cách nghiêm ngặt:\n"
            "1. Trả lời trung thực, khách quan và chính xác dựa trên tài liệu. Không bịa đặt hoặc suy diễn vượt quá tài liệu.\n"
            "2. Nếu tài liệu không có thông tin để trả lời câu hỏi, hãy trả lời rõ ràng rằng bạn không tìm thấy thông tin này trong tài liệu.\n"
            "3. Trích dẫn nguồn cụ thể cho các thông tin quan trọng. Mỗi trích dẫn (citation) cần có đúng số trang (pageNumber), đoạn trích nguyên văn (quote), và thông tin tài liệu (documentId, documentName) nếu có.\n"
            "4. Phản hồi bằng tiếng Việt trôi chảy, rõ ràng. Bạn có thể sử dụng bảng Markdown hoặc danh sách để so sánh/liệt kê thông tin nếu thấy phù hợp.\n"
            "5. Nếu người dùng yêu cầu tạo đề kiểm tra, bài thi hoặc các câu hỏi trắc nghiệm/tự luận ôn tập từ tài liệu, hãy lịch sự từ chối và nhắc họ rằng bạn chỉ tập trung hỗ trợ giải đáp thắc mắc kiến thức dựa trên nội dung tài liệu."
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
                model=settings.gemini_model_name,
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
            logger.error(f"[Synthesis Agent] Error: {e}", exc_info=True)
            return {
                "answer": "Không thể tổng hợp câu trả lời dựa trên tài liệu. Vui lòng thử lại sau.",
                "citations": [],
                "condensedQuestion": question,
                "promptSent": prompt
            }

    def evaluator_agent(self, answer: str, context: List[dict]) -> Tuple[bool, float]:
        logger.info("[Evaluator Agent] Performing Quality Control Check on generated answer.")
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
                model=settings.gemini_evaluator_model_name,
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
            logger.error(f"[Evaluator Agent] Evaluation error: {e}", exc_info=True)
            return True, 1.0


def generate_chat_response(request: ChatRequest) -> ChatResponse:
    logger.info(f"Bắt đầu xử lý câu hỏi (Multi-Agent đồng bộ): {request.question}")
    
    orchestrator = MultiAgentOrchestrator()
    
    # Chuẩn bị dữ liệu cho các Agent
    raw_context = [
        {
            "pageNumber": ctx.pageNumber,
            "text": ctx.text,
            "documentName": ctx.documentName,
            "documentId": ctx.documentId
        } for ctx in request.context
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
    if intent == "RAG" and raw_context:
        filtered_context = orchestrator.retrieval_agent(request.question, raw_context)
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
        
    # Map citations sang DTO Citation
    citations_mapped = []
    for c in citations:
        citations_mapped.append(Citation(
            pageNumber=c.get("pageNumber"),
            quote=c.get("quote", ""),
            documentId=c.get("documentId"),
            documentName=c.get("documentName")
        ))
        
    return ChatResponse(
        answer=answer,
        citations=citations_mapped,
        condensedQuestion=condensed_question,
        promptSent=prompt_sent
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
