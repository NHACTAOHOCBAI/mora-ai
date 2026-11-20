from unittest.mock import patch

from app.services.chat_service import MultiAgentOrchestrator


def test_general_prompt_prefers_direct_answers_without_forced_sections():
    with patch("app.services.chat_service.get_client"), patch(
        "app.services.chat_service.generate", return_value="Câu trả lời"
    ) as generate:
        MultiAgentOrchestrator().general_chat_agent("Thuế là gì?", "", [])

    instruction = generate.call_args.args[3]
    assert "Mở đầu bằng một câu văn xuôi trả lời trực tiếp câu hỏi" in instruction
    assert "Không tạo tiêu đề hoặc danh sách chỉ để trang trí" in instruction
    assert "Chỉ dùng danh sách khi có từ 3 ý độc lập" in instruction
    assert "không dùng “người này” hay “đối tượng này”" in instruction
    assert "highlight cú pháp" not in instruction


def test_rag_prompt_keeps_citations_and_uses_shared_presentation_rules():
    payload = '{"answer":"Có.","citations":[],"hasSufficientInfo":true}'
    with patch("app.services.chat_service.get_client"), patch(
        "app.services.chat_service.generate", return_value=payload
    ) as generate:
        result = MultiAgentOrchestrator().synthesis_agent(
            "Có không?", [{"text": "Có.", "pageNumber": 1, "documentId": 7}], "", []
        )

    instruction = generate.call_args.args[3]
    assert result["hasSufficientInfo"] is True
    assert "TRÍCH DẪN NGUỒN" in instruction
    assert "Chỉ dùng bảng Markdown để so sánh" in instruction
    assert "không dùng cụm này làm câu mở đầu mặc định" in instruction
