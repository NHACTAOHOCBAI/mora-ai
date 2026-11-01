import base64
import io
import pypdf
from app.core.config import settings
from app.services.openai_service import get_client, generate


def extract_page(client, page_pdf_bytes, page_num):
    prompt = (
        "Bạn là chuyên gia AI trích xuất và số hóa tài liệu học thuật cho mạng xã hội học tập Mora. "
        "Hãy trích xuất và chuyển đổi toàn bộ nội dung của trang tài liệu PDF này sang định dạng Markdown chuẩn với các yêu cầu sau:\n"
        "1. Giữ nguyên cấu trúc phân cấp tiêu đề (#, ##, ###), danh sách và các đoạn văn bản.\n"
        "2. Nếu có bảng biểu (Tables), bắt buộc định dạng chuẩn cú pháp Markdown Table (| Cột 1 | Cột 2 |...) với đầy đủ dữ liệu.\n"
        "3. Nếu có hình ảnh, sơ đồ lưu đồ (Flowchart), biểu đồ (Chart), hãy chèn mục `[MÔ TẢ SƠ ĐỒ / HÌNH ẢNH]:` và phân tích chi tiết các thành phần, luồng xử lý và số liệu thống kê.\n"
        "4. Nếu có công thức toán học hoặc ký hiệu khoa học, hãy giữ nguyên định dạng LaTeX hoặc ký tự chuẩn.\n"
        "5. Chỉ trả về nội dung Markdown trích xuất trực tiếp, tuyệt đối không thêm lời chào mở đầu hay kết thúc."
    )

    content = [{"role": "user", "content": [
        {"type": "input_file", "filename": f"page-{page_num}.pdf",
         "file_data": "data:application/pdf;base64," + base64.b64encode(page_pdf_bytes).decode()},
        {"type": "input_text", "text": prompt}
    ]}]
    return generate(client, settings.openai_parser_model, content,
                    "Trích xuất trang PDF thành Markdown chính xác, giữ bảng, công thức và mô tả hình.")


def parse_pdf_layout_and_diagrams(pdf_bytes):
    client = get_client()
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        writer = pypdf.PdfWriter()
        writer.add_page(page)
        stream = io.BytesIO()
        writer.write(stream)
        pages.append({"pageNumber": page_num,
                      "text": extract_page(client, stream.getvalue(), page_num)})
    return pages
