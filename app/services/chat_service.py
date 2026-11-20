import json
import time
from typing import List, Tuple, Optional
from pydantic import BaseModel
from loguru import logger
from app.services.openai_service import get_client, generate, stream_text, AI_ERROR
from app.core.stream_control import finish, is_cancel_requested, start

from app.core.config import settings
from app.core.execution_trace import event, failure, execution_trace
from app.schemas.chat import ChatRequest, ChatResponse, Citation, ChatSummarizeRequest, ChatSummarizeResponse


PRESENTATION_INSTRUCTIONS = """
QUY TẮC TRÌNH BÀY:
- Mở đầu bằng một câu văn xuôi trả lời trực tiếp câu hỏi. Không bắt đầu bằng bullet, heading, “Theo tài liệu”, câu xã giao hoặc việc nhắc lại câu hỏi.
- Với câu hỏi đơn giản hoặc chỉ có 1–2 thông tin bổ sung, trả lời gọn trong 1–3 câu/đoạn ngắn; không tạo heading hoặc danh sách chỉ để trang trí.
- Chỉ dùng danh sách khi có từ 3 ý độc lập cần quét nhanh. Dùng một cấp list; mỗi ý ngắn, có giá trị riêng và có thể dùng nhãn in đậm khi giúp phân biệt trường thông tin. Tránh danh sách lồng nhau trừ khi không thể diễn đạt rõ bằng một cấp.
- Chỉ dùng một tiêu đề Markdown (###) khi câu trả lời có từ hai phần độc lập.
- Chỉ dùng bảng Markdown để so sánh từ hai đối tượng/tiêu chí trở lên. Không dùng bảng thay cho văn xuôi đơn giản.
- Chỉ in đậm các từ khóa thật sự quan trọng; dùng inline code chỉ cho tên kỹ thuật, lệnh, biến hoặc hàm.
- Gọi người/vật bằng tên hoặc vai trò cụ thể trong ngữ cảnh, không dùng “người này” hay “đối tượng này”. Trả lời bằng ngôn ngữ của người dùng; Việt hoá các trạng thái phổ thông, chỉ giữ tiếng Anh cho tên riêng và thuật ngữ chuyên ngành.
- Chỉ nêu “Theo tài liệu/CV” khi cần làm rõ căn cứ, mức độ chắc chắn hoặc giới hạn thông tin; không dùng cụm này làm câu mở đầu mặc định.
- Dùng dòng trống giữa các đoạn/phần. Không dùng quá hai cấp heading và không tạo section “Kết luận” chỉ để lặp lại nội dung vừa nói.
- Với code, dùng fenced code block có nhãn ngôn ngữ khi biết; không khẳng định có syntax highlighting.
"""

# Definitions for structured JSON schema response from agents
class CitationSchema(BaseModel):
    pageNumber: int
    quote: str
    documentId: int | None = None
    documentName: str | None = None

class RAGResponseSchema(BaseModel):
    answer: str
    citations: List[CitationSchema]
    hasSufficientInfo: bool

class RouteSchema(BaseModel):
    intent: str  # "RAG" or "GENERAL"
    reason: str

class EvaluationSchema(BaseModel):
    is_faithful: bool
    hallucinations: List[str]
    score: float  # 0.0 to 1.0

class CitationListSchema(BaseModel):
    citations: List[CitationSchema]

class MultiAgentOrchestrator:
    def __init__(self):
        self.client = get_client()
        self.chat_model = settings.openai_model
        self.router_model = settings.openai_model
        self.evaluator_model = settings.openai_model

    def route_agent(self, question: str, chat_summary: str, history: List[dict]) -> str:
        logger.info(f"[Router Agent] Classifying intent using model '{self.router_model}' for: '{question}'")
        history_str = "\n".join([f"{h.get('sender')}: {h.get('text')}" for h in history[-12:]])
        system_instruction = (
            "Bạn là trợ lý định tuyến (routing agent) cho hệ thống Multi-Agent.\n"
            "Nhiệm vụ của bạn là phân loại xem câu hỏi của người dùng có yêu cầu thông tin từ tài liệu đã tải lên của họ hay đó là một cuộc trò chuyện/yêu cầu chung.\n\n"
            "Quy tắc:\n"
            "1. Mặc định phân loại là 'RAG' cho TẤT CẢ các câu hỏi hỏi về kiến thức, học thuật, khái niệm, lý luận, lý thuyết chuyên ngành (ngay cả khi người dùng không nhắc chữ 'tài liệu' hay 'bài giảng'). RAG là ưu tiên hàng đầu trong môi trường học tập.\n"
            "2. Chỉ phân loại là 'GENERAL' khi câu hỏi thuần túy là giao tiếp xã giao (chào hỏi), hoặc các yêu cầu rõ ràng nằm ngoài việc học như: viết code độc lập, viết email, dịch thuật ngẫu nhiên, hoặc khi người dùng yêu cầu bỏ qua tài liệu.\n"
            "3. Trả về phản hồi theo đúng cấu trúc JSON được yêu cầu."
        )
        prompt = (
            f"--- Lịch sử cuộc trò chuyện ---\n{history_str}\n"
            f"--- Tóm tắt lịch sử trước đó ---\n{chat_summary}\n"
            f"--- Câu hỏi mới ---\n{question}\n"
        )
        try:
            response = generate(self.client, self.router_model, prompt, system_instruction, RouteSchema, operation="chat_router")
            result = json.loads(response)
            intent = result.get("intent", "GENERAL").upper()
            logger.info(f"[Router Agent] Decision: {intent} (Reason: {result.get('reason')})")
            return intent
        except Exception as error:
            failure("router", error)
            raise RuntimeError(AI_ERROR) from None

    def retrieval_agent(
        self, 
        question: str, 
        space_id: Optional[int] = None, 
        raw_context: List[dict] = None,
        document_ids: Optional[List[int]] = None,
        usage_sink: list | None = None
    ) -> List[dict]:
        logger.info(f"[Retrieval Agent] Bắt đầu xử lý truy vấn cho Space ID: {space_id}, Document IDs: {document_ids}")
        if space_id and settings.retrieval_enabled and document_ids != []:
            try:
                from app.services.retrieval_service import hybrid_retrieve
                retrieved_chunks = hybrid_retrieve(
                    query=question,
                    space_id=space_id,
                    openai_client=self.client,
                    document_ids=document_ids,
                    usage_sink=usage_sink
                )
                if retrieved_chunks:
                    event("retrieval", contextSource="hybrid")
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
            except Exception as error:
                failure("retrieval", error)
                raise RuntimeError(AI_ERROR) from None

        logger.info(f"[Retrieval Agent] Sử dụng raw_context đầu vào. Tổng số: {len(raw_context) if raw_context else 0}")
        event("retrieval", contextSource="raw_context")
        return raw_context or []

    def general_chat_agent(self, question: str, chat_summary: str, history: List[dict]) -> Tuple[str, str]:
        logger.info(f"[General Chat Agent] Answering query using model '{self.chat_model}': '{question}'")
        system_instruction = (
            "Bạn là Trợ lý Học tập AI tích hợp trong hệ thống Mora.\n"
            "Nhiệm vụ của bạn là trợ giúp người dùng giải quyết các câu hỏi học thuật chung (như giải thích lý thuyết, viết code, giải toán phổ thông, dịch thuật...).\n\n"
            + PRESENTATION_INSTRUCTIONS + "\n"
            "Nếu người dùng yêu cầu tạo đề kiểm tra, bài thi hoặc các câu hỏi trắc nghiệm/tự luận, hãy lịch sự từ chối và nhắc họ rằng bạn chỉ tập trung hỗ trợ giải đáp thắc mắc kiến thức.\n"
            "Chỉ nhắc người dùng đây là chế độ chat tự do khi họ cố tình hỏi thông tin nằm trong tài liệu mà hệ thống thiếu context; các trường hợp khác trả lời tự nhiên."
        )
        if chat_summary:
            system_instruction += f"\nTóm tắt lịch sử hội thoại trước đó: {chat_summary}"
        history_str = "\n".join([f"{h.get('sender')}: {h.get('text')}" for h in history[-12:]])
        prompt = (
            f"--- LỊCH SỬ HỘI THOẠI ---\n{history_str}\n"
            f"--- CÂU HỎI MỚI ---\nNgười dùng: {question}\n"
        )
        try:
            response = generate(self.client, self.chat_model, prompt, system_instruction, operation="chat_generation")
            return response, system_instruction + "\n\n" + prompt
        except Exception as error:
            failure("general", error)
            raise RuntimeError(AI_ERROR) from None

    def synthesis_agent(self, question: str, context: List[dict], chat_summary: str, history: List[dict]) -> dict:
        logger.info(f"[Synthesis Agent] Synthesizing answer using model '{self.chat_model}' with {len(context)} chunks.")
        context_str = ""
        for item in context:
            doc_name = item.get("documentName", f"Tài liệu #{item.get('documentId')}")
            context_str += f"Tài liệu: {doc_name} (ID: {item.get('documentId')}) - Trang {item.get('pageNumber')}\nNội dung:\n{item.get('text')}\n---\n"
        history_str = "\n".join([f"{h.get('sender')}: {h.get('text')}" for h in history[-12:]])
        system_instruction = (
            "Bạn là Trợ lý Học tập AI tích hợp trong hệ thống Mora (Source-Grounded AI Learning Assistant).\n"
            "Nhiệm vụ của bạn là trả lời các câu hỏi học thuật từ người dùng dựa trên ngữ cảnh tài liệu được cung cấp phía dưới.\n\n"
            "HÃY TUÂN THỦ CÁC QUY TẮC SAU MỘT CÁCH NGHIÊM NGẶT:\n"
            "1. TÍNH TRUNG THỰC & CHÍNH XÁC: Trả lời trung thực, khách quan và chính xác dựa trên tài liệu. Không bịa đặt hoặc suy diễn vượt quá tài liệu.\n"
            "2. THIẾU THÔNG TIN: Trả về 'hasSufficientInfo': true nếu tài liệu có ĐỦ thông tin. Nếu tài liệu KHÔNG CÓ đủ thông tin để trả lời trọn vẹn câu hỏi, hãy đặt 'hasSufficientInfo': false và để trống trường 'answer'.\n"
            "3. TRÍCH DẪN NGUỒN (CITATIONS): Trích dẫn nguồn cụ thể cho các thông tin quan trọng. Mỗi trích dẫn (citation) cần có đúng số trang (pageNumber), đoạn trích nguyên văn (quote), và thông tin tài liệu (documentId, documentName) nếu có.\n"
            "4. " + PRESENTATION_INSTRUCTIONS + "\n"
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
            response = generate(self.client, self.chat_model, prompt, system_instruction, RAGResponseSchema, operation="rag_generation")
            result = json.loads(response)
            result["promptSent"] = system_instruction + "\n\n" + prompt
            return result
        except Exception as error:
            failure("synthesis", error)
            raise RuntimeError(AI_ERROR) from None

    def evaluator_agent(self, answer: str, context: List[dict]) -> Tuple[bool, Optional[float]]:
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
            response = generate(self.client, self.evaluator_model, prompt, system_instruction, EvaluationSchema, operation="rag_evaluation")
            result = EvaluationSchema.model_validate_json(response)
            is_faithful = result.is_faithful
            score = result.score
            logger.info(f"[Evaluator Agent] Quality score: {score} (Is Faithful: {is_faithful})")
            return is_faithful, score
        except Exception as error:
            failure("evaluator", error)
            if execution_trace.get() is not None:
                return False, None
            raise RuntimeError(AI_ERROR) from None


def generate_chat_response(request: ChatRequest) -> ChatResponse:
    t_start = time.perf_counter()
    orchestrator = MultiAgentOrchestrator()

    space_id = request.space_id or request.spaceId
    document_ids = request.document_ids if request.document_ids is not None else request.documentIds
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
    t_router_start = time.perf_counter()
    intent = orchestrator.route_agent(request.question, request.chat_summary or "", history)
    t_router = time.perf_counter() - t_router_start
    
    answer = ""
    citations = []
    t_retrieval = 0.0
    t_synthesis_total = 0.0
    t_eval_total = 0.0
    attempts_count = 0
    
    # 2. Xử lý theo phân loại
    is_rag_success = False
    qc_unavailable = False
    qc_status = "not_run"
    filtered_context = []

    if intent == "RAG":
        t_retrieval_start = time.perf_counter()
        filtered_context = orchestrator.retrieval_agent(
            question=request.question, 
            space_id=space_id, 
            raw_context=raw_context,
            document_ids=document_ids
        )
        t_retrieval = time.perf_counter() - t_retrieval_start
        
        if not filtered_context:
            logger.warning(f"[Orchestrator] Không tìm thấy ngữ cảnh nào cho câu hỏi RAG: '{request.question}'. Sẽ chuyển sang GENERAL fallback.")
        else:
            max_retries = 2
            for attempt in range(max_retries):
                attempts_count = attempt + 1
                logger.info(f"[Orchestrator] Synthesis attempt {attempt + 1}")
                
                t_synth_start = time.perf_counter()
                rag_result = orchestrator.synthesis_agent(request.question, filtered_context, request.chat_summary or "", history)
                t_synth = time.perf_counter() - t_synth_start
                t_synthesis_total += t_synth
                
                has_info = rag_result.get("hasSufficientInfo", True)
                if not has_info:
                    logger.warning("[Orchestrator] Document lacks sufficient info (hasSufficientInfo=False). Triggering fallback.")
                    break
                
                answer = rag_result.get("answer", "")
                citations = rag_result.get("citations", [])
                
                # Evaluator Agent kiểm QC câu trả lời
                t_eval_start = time.perf_counter()
                is_faithful, score = orchestrator.evaluator_agent(answer, filtered_context)
                t_eval = time.perf_counter() - t_eval_start
                t_eval_total += t_eval
                
                if score is None:
                    qc_unavailable = True
                    qc_status = "error"
                    break
                if is_faithful or score >= 0.7:
                    qc_status = "passed"
                    logger.info(f"[Orchestrator] QC passed! (Score: {score}, Synthesis: {t_synth:.2f}s, Evaluator: {t_eval:.2f}s)")
                    is_rag_success = True
                    break
                else:
                    qc_status = "failed"
                    logger.warning(f"[Orchestrator] QC failed with score {score} (Synthesis: {t_synth:.2f}s, Evaluator: {t_eval:.2f}s). Retrying synthesis...")

    # Nếu intent là GENERAL, hoặc RAG fail (không có context, hoặc thiếu info)
    if intent == "GENERAL" or (intent == "RAG" and not is_rag_success and not qc_unavailable):
        logger.info("[Orchestrator] Executing General Chat Agent (either intended or via RAG Fallback)")
        t_synth_start = time.perf_counter()
        general_answer, prompt_sent = orchestrator.general_chat_agent(request.question, request.chat_summary or "", history)
        t_synthesis_total += time.perf_counter() - t_synth_start
        
        if intent == "RAG":
            # Tiền tố thông báo fallback
            answer = "⚠️ *Tài liệu học tập không chứa thông tin để trả lời câu hỏi này. Tuy nhiên, dựa trên kiến thức chung của AI:*\n\n" + general_answer
        else:
            answer = general_answer
            
        citations = []

    t_total = time.perf_counter() - t_start
    if intent == "RAG":
        logger.info(
            f"⚡ [Latency Metrics] Tổng thời gian phản hồi: {t_total:.2f}s "
            f"(Router: {t_router:.2f}s | Retrieval: {t_retrieval:.2f}s | Synthesis ({attempts_count}x): {t_synthesis_total:.2f}s | Evaluator: {t_eval_total:.2f}s)"
        )
    else:
        logger.info(
            f"⚡ [Latency Metrics] Tổng thời gian phản hồi (General Chat): {t_total:.2f}s "
            f"(Router: {t_router:.2f}s | Synthesis: {t_synthesis_total:.2f}s)"
        )
        
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
        
    event("pipeline", intent=intent, fallback=intent == "RAG" and not is_rag_success and not qc_unavailable,
          qcStatus=qc_status, contexts=filtered_context, synthesisAttempts=attempts_count,
          timings={"totalMs": (time.perf_counter() - t_start) * 1000,
                   "routerMs": t_router * 1000, "retrievalMs": t_retrieval * 1000,
                   "synthesisMs": t_synthesis_total * 1000, "evaluatorMs": t_eval_total * 1000})
    return ChatResponse(
        answer=answer,
        citations=citations_mapped,
    )

def generate_chat_response_stream(request: ChatRequest, usage_sink: list | None = None):
    """Run routing/retrieval first, then yield answer text as OpenAI generates it.

    Citations are extracted after text generation so they never block the first token.
    """
    orchestrator = MultiAgentOrchestrator()
    stream_control = start(request.stream_id, orchestrator.client) if request.stream_id else None
    history = [{"sender": h.sender, "text": h.text} for h in request.history]
    space_id = request.space_id or request.spaceId
    document_ids = request.document_ids if request.document_ids is not None else request.documentIds
    raw_context = [{"pageNumber": c.pageNumber, "text": c.text, "documentName": c.documentName,
                    "documentId": c.documentId, "sectionPath": c.sectionPath} for c in (request.context or [])]
    # Streaming must not wait for a separate router-model completion.  In a Space,
    # an explicit ready-document scope is the user's intent to ask the documents.
    context = []
    intent = "RAG" if document_ids else "GENERAL"
    if intent == "RAG":
        yield "status", {"stage": "retrieving"}
        context = orchestrator.retrieval_agent(request.question, space_id, raw_context, document_ids,
                                               usage_sink=usage_sink)

    if intent == "RAG" and context:
        context_str = "\n---\n".join(
            f"Tài liệu: {c.get('documentName') or c.get('documentId')} | Trang: {c.get('pageNumber')}\n{c.get('text', '')}"
            for c in context)
        instructions = ("Bạn là trợ lý học tập Mora. Chỉ trả lời bằng thông tin được hỗ trợ bởi tài liệu. "
                        "Nếu thiếu dữ kiện, nói rõ phần nào không có trong tài liệu. Dùng Markdown rõ ràng.")
        prompt = f"--- TÀI LIỆU ---\n{context_str}\n--- LỊCH SỬ ---\n" + \
                 "\n".join(f"{h['sender']}: {h['text']}" for h in history[-12:]) + \
                 f"\n--- CÂU HỎI ---\n{request.question}"
    else:
        if intent == "RAG":
            prefix = "⚠️ *Tài liệu học tập không chứa thông tin để trả lời câu hỏi này. Tuy nhiên, dựa trên kiến thức chung của AI:*\n\n"
        else:
            prefix = ""
        instructions = ("Bạn là Trợ lý Học tập AI Mora. Trả lời tự nhiên, chính xác, dùng Markdown rõ ràng "
                        "với tiêu đề và danh sách khi phù hợp.")
        prompt = "--- LỊCH SỬ ---\n" + "\n".join(f"{h['sender']}: {h['text']}" for h in history[-12:]) + f"\n--- CÂU HỎI ---\n{request.question}"
        if prefix:
            yield "delta", {"text": prefix}

    yield "status", {"stage": "generating"}
    answer_parts = []
    stream_operation = "rag_generation" if intent == "RAG" and context else "chat_generation"
    for delta in stream_text(orchestrator.client, orchestrator.chat_model, prompt, instructions,
                             stream_operation, usage_sink=usage_sink, stream_control=stream_control):
        answer_parts.append(delta)
        yield "delta", {"text": delta}
    answer = "".join(answer_parts)

    citations = []
    if not is_cancel_requested(stream_control) and intent == "RAG" and context and answer.strip():
        yield "status", {"stage": "citing"}
        citation_prompt = (f"Câu trả lời:\n{answer}\n\nNgữ cảnh:\n" +
                           "\n---\n".join(json.dumps(c, ensure_ascii=False) for c in context) +
                           "\n\nChỉ trả các trích dẫn trực tiếp hỗ trợ câu trả lời.")
        citation_instruction = "Trích xuất citations chính xác từ ngữ cảnh. Không bịa quote, trang hoặc tài liệu."
        try:
            result = json.loads(generate(orchestrator.client, orchestrator.chat_model, citation_prompt,
                                         citation_instruction, CitationListSchema, operation="rag_citations",
                                         usage_sink=usage_sink))
            citations = result.get("citations", [])
        except Exception as error:
            failure("citations", error)
    if not is_cancel_requested(stream_control):
        yield "citations", {"citations": citations}
    yield "usage", {"usage": usage_sink or []}
    if request.stream_id:
        finish(request.stream_id, stream_control)

def generate_chat_summary(request: ChatSummarizeRequest) -> ChatSummarizeResponse:
    client = get_client()
    model_name = settings.openai_model

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
        response = generate(client, model_name, prompt, system_instruction, operation="chat_summary")
        summary = response.strip()
        logger.info(f"Tóm tắt hội thoại thành công: {summary}")
        return ChatSummarizeResponse(summary=summary)
    except Exception:
        raise RuntimeError(AI_ERROR) from None
