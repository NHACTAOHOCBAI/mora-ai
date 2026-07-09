import io
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from collections import defaultdict
from loguru import logger
from google import genai
from google.genai import types
from app.core.config import settings

# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, DocumentStream
from docling_core.types.doc import PictureItem

def describe_image_with_gemini(client: genai.Client, image_bytes: bytes, ext: str, page_num: int, source_type: str) -> str:
    # Map extension to mime type
    mime_type = "image/png"
    if ext.lower() in ["jpg", "jpeg"]:
        mime_type = "image/jpeg"
    elif ext.lower() == "webp":
        mime_type = "image/webp"

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )

    prompt = (
        "Đây là hình ảnh hoặc sơ đồ được trích xuất từ một trang tài liệu PDF. "
        "Hãy phân tích và mô tả chi tiết sơ đồ, biểu đồ hoặc hình vẽ này dưới dạng văn bản tiếng Việt để làm tài liệu tra cứu. "
        "Nếu là sơ đồ/lưu đồ, hãy nêu rõ các thành phần, các bước và luồng xử lý. "
        "Nếu là biểu đồ, hãy nêu rõ các thông số và số liệu thống kê cốt lõi. "
        "Nếu là hình vẽ minh họa, hãy mô tả chi tiết đối tượng vẽ và ý nghĩa của nó."
    )

    try:
        logger.info(f"[Trang {page_num}] Đang gửi ảnh chụp {source_type} ({len(image_bytes)} bytes) sang Gemini Vision...")
        response = client.models.generate_content(
            model=settings.gemini_model_name,
            contents=[image_part, prompt]
        )
        caption = response.text
        logger.info(f"[Trang {page_num}] Nhận phản hồi mô tả sơ đồ từ Gemini Vision thành công (dài {len(caption)} ký tự).")
        return caption
    except Exception as e:
        logger.error(f"[Trang {page_num}] Lỗi khi gọi Gemini Vision để mô tả {source_type}: {e}")
        return ""

def parse_pdf_layout_and_diagrams(pdf_bytes: bytes) -> list:
    logger.info("========================================= MORA DOCLING PARSING START =========================================")
    logger.info(f"Bắt đầu phân tích cấu trúc PDF bằng IBM Docling. Kích thước file: {len(pdf_bytes)} bytes")
    
    client = genai.Client(api_key=settings.gemini_api_key)
    parsed_pages = []

    try:
        # Cấu hình Docling Pipeline
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True  # Trích xuất hình ảnh
        pipeline_options.images_scale = 2.0  # Tăng độ phân giải cho ảnh trích xuất

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        # Đọc PDF từ memory stream
        pdf_stream = io.BytesIO(pdf_bytes)
        doc_stream = DocumentStream(name="document.pdf", stream=pdf_stream)
        
        logger.info("Đang chuyển đổi tài liệu bằng Docling...")
        conv_res = doc_converter.convert(doc_stream)
        doc = conv_res.document
        total_pages = len(doc.pages)
        logger.info(f"Phân tích tài liệu thành công. Tổng số trang: {total_pages}")

        # Gom nhóm các bức ảnh (PictureItem) theo trang (page_no)
        pictures_by_page = defaultdict(list)
        for element, _level in doc.iterate_items():
            if isinstance(element, PictureItem):
                if element.prov and len(element.prov) > 0:
                    page_no = element.prov[0].page_no
                    pictures_by_page[page_no].append(element)

        # Xử lý nội dung từng trang
        for page_num in range(1, total_pages + 1):
            logger.info(f"[Trang {page_num}/{total_pages}] Đang trích xuất văn bản & bảng biểu Markdown...")
            
            # Trích xuất Markdown thô của trang (bao gồm cả Tables đã chuyển sang Markdown tự động bởi Docling)
            page_text = doc.export_to_markdown(page_no=page_num, image_mode="placeholder")
            page_text = page_text.strip()

            # Lấy và mô tả tất cả ảnh thuộc trang này
            page_pictures = pictures_by_page[page_num]
            if page_pictures:
                logger.info(f"[Trang {page_num}] Phát hiện {len(page_pictures)} ảnh/sơ đồ trích xuất từ Docling.")
                for img_idx, element in enumerate(page_pictures):
                    try:
                        # Lấy PIL Image của ảnh từ tài liệu
                        pil_img = element.get_image(doc)
                        if pil_img:
                            # Convert PIL Image to bytes
                            img_byte_arr = io.BytesIO()
                            pil_img.save(img_byte_arr, format='PNG')
                            img_bytes = img_byte_arr.getvalue()

                            # Gọi Gemini mô tả hình ảnh
                            image_desc = describe_image_with_gemini(
                                client, 
                                img_bytes, 
                                "png", 
                                page_num, 
                                f"Ảnh trích xuất #{img_idx + 1}"
                            )
                            if image_desc:
                                page_text += f"\n\n[MÔ TẢ HÌNH ẢNH TRÊN TRANG {page_num}]:\n{image_desc.strip()}\n\n"
                    except Exception as img_err:
                        logger.error(f"[Trang {page_num}] Lỗi khi xử lý ảnh trích xuất #{img_idx + 1}: {img_err}")

            parsed_pages.append({
                "pageNumber": page_num,
                "text": page_text
            })
            logger.info(f"[Trang {page_num}] Hoàn thành xử lý trang.")

    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng trong quá trình Docling parsing: {e}")
        raise e

    logger.info("========================================= MORA DOCLING PARSING END =========================================")
    return parsed_pages
