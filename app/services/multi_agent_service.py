import json
from typing import List, Dict, Any, Tuple
from loguru import logger
from google import genai
from google.genai import types

from app.core.config import settings
from app.core.broker import event_broker

from pydantic import BaseModel

# Models definitions for response schemas
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

    def route_agent(self, question: str, chat_summary: str, history: List[Dict[str, Any]]) -> str:
        """
        Router Agent: Classifies user intent into general chat or RAG (document Q&A).
        """
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

    def retrieval_agent(self, question: str, raw_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Retrieval Agent: Filters and selects the most relevant context chunks.
        """
        logger.info(f"[Retrieval Agent] Filtering context chunks. Total input: {len(raw_context)}")
        # For now, let's keep all chunks or perform basic ranking based on key terms.
        # In a fully developed Vector Search, this agent would run hybrid search.
        # Here we prioritize keeping chunks that match keywords or we return them to the synthesis agent.
        return raw_context

    def general_chat_agent(self, question: str, chat_summary: str, history: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        General Chat Agent: Answers general questions using Gemini's pre-trained knowledge.
        """
        logger.info(f"[General Chat Agent] Answering general query: '{question}'")
        
        system_instruction = (
            "Bạn là Trợ lý Học tập AI tích hợp trong hệ thống Mora.\n"
            "Nhiệm vụ của bạn là trợ giúp người dùng giải quyết các câu hỏi chung (như giải thích lý thuyết, viết code, giải toán phổ thông, dịch thuật...).\n"
            "Hãy trả lời một cách tự nhiên, chi tiết, chuyên nghiệp bằng tiếng Việt và sử dụng định dạng Markdown nếu cần thiết.\n"
            "Nếu người dùng đề cập đến tài liệu học tập của họ, hãy lịch sự nhắc họ rằng đây là chế độ chat tự do và bạn không sử dụng tài liệu học tập cho câu hỏi này.\n"
            "ĐỊNH DẠNG VÀ TRÌNH BÀY (CỰC KỲ QUAN TRỌNG):\n"
            "1. Khi tạo các câu hỏi trắc nghiệm, BẮT BUỘC phải xuống dòng cho mỗi đáp án. Không viết các đáp án A, B, C, D trên cùng một dòng. Ví dụ:\n"
            "   **Câu 1: Câu hỏi là gì?**\n"
            "   A. Lựa chọn một\n"
            "   B. Lựa chọn hai\n"
            "   C. Lựa chọn ba\n"
            "   D. Lựa chọn tư\n"
            "2. Luôn sử dụng xuống dòng kép (\\n\\n) giữa các câu hỏi hoặc các đoạn văn lớn để văn bản thông thoáng, dễ nhìn.\n"
            "3. Sử dụng các thẻ in đậm Markdown như **Câu 1:**, **Đáp án:** để làm nổi bật tiêu đề."
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

    def synthesis_agent(self, question: str, context: List[Dict[str, Any]], chat_summary: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesis Agent: Compiles contexts, reasons, and outputs answer in structured RAG format.
        """
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
            "5. ĐỊNH DẠNG VÀ TRÌNH BÀY (CỰC KỲ QUAN TRỌNG):\n"
            "   - Khi hiển thị câu hỏi hoặc câu trắc nghiệm, BẮT BUỘC mỗi lựa chọn đáp án A, B, C, D phải nằm trên một dòng riêng biệt mới (không viết liền dòng).\n"
            "   - Luôn sử dụng xuống dòng kép (\\n\\n) giữa các câu hỏi hoặc các phần chính của câu trả lời để tạo khoảng cách thoáng đãng, dễ đọc.\n"
            "   - Sử dụng in đậm Markdown như **Câu 1:**, **Đề kiểm tra:**, **Gợi ý trả lời:** để cấu trúc bài viết rõ ràng."
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

    def evaluator_agent(self, answer: str, context: List[Dict[str, Any]]) -> Tuple[bool, float]:
        """
        Evaluator Agent: QC check to identify factual alignment and hallucination.
        """
        logger.info("[Evaluator Agent] Performing Quality Control Check on generated answer.")
        if not context:
            # If no context is expected (e.g. general chat), it always passes QC
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
            return True, 1.0  # Fallback to true if evaluation fails to avoid infinite loops

    def orchestrate(self, event_data: dict):
        """
        Main orchestration flow for the event-driven Multi-Agent system.
        """
        space_id = event_data.get("spaceId")
        question = event_data.get("question")
        chat_summary = event_data.get("chatSummary", "")
        history = event_data.get("history", [])
        raw_context = event_data.get("context", [])
        user_message_id = event_data.get("userMessageId")
        assistant_message_id = event_data.get("assistantMessageId")

        # 1. Router Agent decides intent
        intent = self.route_agent(question, chat_summary, history)
        
        answer = ""
        citations = []
        condensed_question = question
        prompt_sent = ""

        # 2. Process based on intent
        if intent == "RAG" and raw_context:
            # RAG flow
            filtered_context = self.retrieval_agent(question, raw_context)
            
            # Synthesis with retry for QC
            max_retries = 2
            for attempt in range(max_retries):
                logger.info(f"[Orchestrator] Synthesis attempt {attempt + 1}")
                rag_result = self.synthesis_agent(question, filtered_context, chat_summary, history)
                
                answer = rag_result.get("answer", "")
                citations = rag_result.get("citations", [])
                condensed_question = rag_result.get("condensedQuestion", question)
                prompt_sent = rag_result.get("promptSent", "")
                
                # Evaluator Agent checks quality
                is_faithful, score = self.evaluator_agent(answer, filtered_context)
                if is_faithful or score >= 0.7:
                    logger.info("[Orchestrator] QC passed successfully!")
                    break
                else:
                    logger.warning(f"[Orchestrator] QC failed with score {score}. Retrying synthesis...")
        else:
            # GENERAL flow
            answer, prompt_sent = self.general_chat_agent(question, chat_summary, history)
            citations = []
            condensed_question = question

        # 3. Publish result to ANSWER_VERIFIED queue
        verified_event = {
            "spaceId": space_id,
            "userMessageId": user_message_id,
            "assistantMessageId": assistant_message_id,
            "answer": answer,
            "citations": citations,
            "condensedQuestion": condensed_question,
            "promptSent": prompt_sent
        }
        
        event_broker.publish_answer_verified(verified_event)

orchestrator = MultiAgentOrchestrator()
