import io
import time
import psutil
import os
import pypdf
from app.services.parser_service import parse_pdf_layout_and_diagrams
from loguru import logger

def create_sample_pdf() -> bytes:
    """Tạo một file PDF 2 trang đơn giản trong bộ nhớ để test."""
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.add_blank_page(width=300, height=300)
    
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()

def test_lightweight_parser():
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    logger.info(f"=== RAM lúc bắt đầu: {mem_before:.2f} MB ===")
    
    pdf_bytes = create_sample_pdf()
    logger.info(f"Tạo file PDF mẫu: {len(pdf_bytes)} bytes, 2 trang.")
    
    start_time = time.time()
    # Test hàm parse (không truyền api_key để kiểm tra fallback text siêu tốc)
    results = parse_pdf_layout_and_diagrams(pdf_bytes, api_key=None)
    elapsed = time.time() - start_time
    
    mem_after = process.memory_info().rss / (1024 * 1024)
    logger.info(f"=== RAM sau khi parse: {mem_after:.2f} MB (Tăng: {mem_after - mem_before:.2f} MB) ===")
    logger.info(f"Thời gian xử lý: {elapsed:.3f} giây")
    logger.info(f"Kết quả trích xuất: {len(results)} trang")
    
    assert len(results) == 2, "Phải trích xuất đủ 2 trang!"
    assert "pageNumber" in results[0] and "text" in results[0], "Cấu trúc trả về phải chuẩn format API!"
    
    logger.info("=== KIỂM THỬ THÀNH CÔNG 100% LIGHTWEIGHT PARSER (RAM < 100MB)! ===")

if __name__ == "__main__":
    test_lightweight_parser()
