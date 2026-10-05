import io
import time
from typing import List, Dict, Any, Optional
from loguru import logger
import pypdf
from google import genai
from google.genai import types
from app.core.config import settings

def extract_page_with_gemini_vision(
    client: genai.Client,
    page_pdf_bytes: bytes,
    page_num: int,
    raw_fallback_text: str = "",
    parser_model: Optional[str] = None
) -> str:
    """Sử dụng Gemini Native Multimodal PDF để đọc và chuyển đổi một trang tài liệu sang Markdown chuẩn (kèm bảng và sơ đồ)."""
    page_part = types.Part.from_bytes(
        data=page_pdf_bytes,
        mime_type="application/pdf"
    )

    prompt = (
        "Bạn là chuyên gia AI trích xuất và số hóa tài liệu học thuật cho mạng xã hội học tập Mora. "
        "Hãy trích xuất và chuyển đổi toàn bộ nội dung của trang tài liệu PDF này sang định dạng Markdown chuẩn với các yêu cầu sau:\n"
        "1. Giữ nguyên cấu trúc phân cấp tiêu đề (#, ##, ###), danh sách và các đoạn văn bản.\n"
        "2. Nếu có bảng biểu (Tables), bắt buộc định dạng chuẩn cú pháp Markdown Table (| Cột 1 | Cột 2 |...) với đầy đủ dữ liệu.\n"
        "3. Nếu có hình ảnh, sơ đồ lưu đồ (Flowchart), biểu đồ (Chart), hãy chèn mục `[MÔ TẢ SƠ ĐỒ / HÌNH ẢNH]:` và phân tích chi tiết các thành phần, luồng xử lý và số liệu thống kê.\n"
        "4. Nếu có công thức toán học hoặc ký hiệu khoa học, hãy giữ nguyên định dạng LaTeX hoặc ký tự chuẩn.\n"
        "5. Chỉ trả về nội dung Markdown trích xuất trực tiếp, tuyệt đối không thêm lời chào mở đầu hay kết thúc."
    )

    if parser_model:
        model_list = [parser_model.strip()]
    else:
        model_list = [m.strip() for m in settings.gemini_parser_model_list.split(",") if m.strip()]
        if not model_list:
            model_list = [settings.gemini_model_name]

    last_error = None
    for idx, model_name in enumerate(model_list):
        for attempt in range(3):
            try:
                logger.info(f"[Trang {page_num}] Đang phân tích layout & sơ đồ qua Gemini Multimodal ({model_name})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[page_part, prompt]
                )
                if response and response.text:
                    parsed_markdown = response.text.strip()
                    logger.info(f"[Trang {page_num}] Gemini trích xuất Markdown thành công ({len(parsed_markdown)} ký tự).")
                    return parsed_markdown
                break
            except Exception as e:
                last_error = e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    logger.warning(f"[Trang {page_num}] Gặp Rate Limit (429) với {model_name}. Đang chờ 3s (thử lại lần {attempt+1})...")
                    time.sleep(3)
                else:
                    logger.warning(f"[Trang {page_num}] Lỗi với model {model_name}: {e}. Đang thử model tiếp theo...")
                    break

    logger.error(f"[Trang {page_num}] Không thể phân tích qua Gemini Vision (Lỗi: {last_error}). Sử dụng text fallback trích xuất từ pypdf.")
    return raw_fallback_text


def parse_pdf_layout_and_diagrams(pdf_bytes: bytes, api_key: Optional[str] = None, parser_model: Optional[str] = None) -> List[Dict[str, Any]]:
    """Trích xuất cấu trúc văn bản, bảng biểu và sơ đồ từ file PDF bằng pypdf kết hợp Gemini Native Multimodal."""
    logger.info("========================================= MORA LIGHTWEIGHT PARSING START =========================================")
    logger.info(f"Bắt đầu phân tích PDF siêu nhẹ (pypdf + Gemini Multimodal). Kích thước file: {len(pdf_bytes)} bytes")

    effective_api_key = api_key or settings.gemini_api_key
    client = None
    if effective_api_key:
        client = genai.Client(api_key=effective_api_key)

    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        logger.info(f"Đọc thành công file PDF với tổng số trang: {total_pages}")
    except Exception as e:
        logger.error(f"Lỗi khi đọc file PDF bằng pypdf: {e}", exc_info=True)
        raise RuntimeError(f"Không thể đọc định dạng PDF: {str(e)}")

    parsed_pages = []

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = reader.pages[page_idx]

        # 1. Trích xuất text thô bằng pypdf
        try:
            raw_text = page.extract_text() or ""
            raw_text = raw_text.strip()
        except Exception as ex:
            logger.warning(f"[Trang {page_num}] Không thể trích xuất text thuần: {ex}")
            raw_text = ""

        # 2. Nếu có Gemini API Client, gửi từng trang PDF slice sang Gemini Multimodal để tái tạo Markdown chuẩn
        if client:
            try:
                # Tách riêng trang này thành 1 file PDF mini trong bộ nhớ
                writer = pypdf.PdfWriter()
                writer.add_page(page)
                page_stream = io.BytesIO()
                writer.write(page_stream)
                page_pdf_bytes = page_stream.getvalue()

                # Gửi sang Gemini Vision
                page_markdown = extract_page_with_gemini_vision(
                    client=client,
                    page_pdf_bytes=page_pdf_bytes,
                    page_num=page_num,
                    raw_fallback_text=raw_text,
                    parser_model=parser_model
                )
            except Exception as e:
                logger.error(f"[Trang {page_num}] Lỗi trong quá trình tạo page slice: {e}. Dùng raw text fallback.")
                page_markdown = raw_text
        else:
            page_markdown = raw_text

        parsed_pages.append({
            "pageNumber": page_num,
            "text": page_markdown
        })
        logger.info(f"[Trang {page_num}/{total_pages}] Hoàn tất xử lý trang.")

    logger.info("========================================= MORA LIGHTWEIGHT PARSING END =========================================")
    return parsed_pages
