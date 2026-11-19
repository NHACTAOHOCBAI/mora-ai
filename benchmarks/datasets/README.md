# Bộ câu hỏi nháp: Hệ điều hành v1

`operating-systems-v1.json` chứa 36 câu hỏi tiếng Việt: **12 DIRECT**, **12 SYNTHESIS**, **12 VISUAL**. Tất cả có `approved: false`. Đây là ground truth **dự thảo cần admin kiểm tra**, không phải bộ chuẩn đã được người có chuyên môn phê duyệt.

## PDF gốc và quy ước trang

PDF nằm trong `../../../mora-data` tính từ thư mục README này. JSON lưu SHA-256 của toàn bộ byte PDF và số trang; không lưu bản sao PDF trong Git.

| Mã | Tệp gốc | Số trang PDF | Số slide in trên tài liệu |
| --- | --- | ---: | --- |
| OS01 | `[BHT CNPM] HDH _Chuong1-1.pdf` | 10 | 1–10 |
| OS02 | `[BHT CNPM] HDH _Chuong1-2.pdf` | 10 | 11–20 |
| OS03 | `[BHT CNPM] HDH _Chuong1-3.pdf` | 10 | 21–30 |
| OS04 | `[BHT CNPM] HDH _Chuong1-4.pdf` | 10 | 31–40 |

**`pageNumber` là vị trí trang trong từng PDF, bắt đầu từ 1.** Ví dụ sơ đồ NUMA nằm ở `OS04`, trang PDF **9**, dù số in trên slide là **39**. Không dùng số slide in để chấm citation.

## Cách soạn và kiểm tra

- Đã đọc text của toàn bộ 40 trang và dựng ảnh từ PDF gốc để xem các hình ở OS01 trang 7, 10; OS02 trang 4, 8; OS03 trang 2, 6, 9; OS04 trang 7, 8, 9.
- Nhóm visual hỏi vị trí khối, mũi tên, đường nối, ranh giới cache/bộ nhớ, nhánh trong lưu đồ và quan hệ theo thời gian. Câu trả lời cần nội dung hình, không chỉ tiêu đề hoặc đoạn văn bên cạnh.
- Mỗi câu có `referenceAnswer`, `explanation` và `sources: [{documentCode, pageNumber}]`. Khi có nhiều nguồn, chúng là tập trang hỗ trợ đáp án, không phải thông tin lọc retrieval.
- Một hình có thể phục vụ hai câu đọc các quan hệ khác nhau. Các câu này không phải mẫu thống kê độc lập hoàn toàn; báo cáo theo nhóm là mô tả baseline của bộ tài liệu nhỏ.
- Các câu chỉ yêu cầu kiến thức trong 4 PDF. Không gửi đáp án, giải thích hoặc `sources` cho pipeline trả lời. Retrieval được tìm trong cả bốn tài liệu.
- `spaceId` và `documentId` ban đầu để `null`. Khi nhập vào admin, tạo/chọn space benchmark riêng, upload đúng PDF qua luồng hiện có, đối chiếu hash và gán mã tài liệu với document ID thực tế; đợi index xong.

Admin cần mở **PDF gốc** tại từng trang nguồn, kiểm tra nội dung câu hỏi, đáp án và sự đầy đủ của tập trang. Với câu visual phải xem hình; không duyệt chỉ dựa trên nội dung parser trả về. Sau khi sửa và duyệt đủ câu, mới chạy benchmark chính thức.

## Kiểm tra offline

Từ thư mục `mora-ai` trên Windows:

```powershell
.\venv\Scripts\python.exe benchmarks\validate_dataset.py
```

Script xác minh số lượng và loại câu hỏi, mã không trùng, trường bắt buộc, khoảng trang hợp lệ, SHA-256 và số trang của bốn PDF. Nó **không thay thế duyệt đúng/sai về nội dung**. Không cần OpenAI API, Qdrant hoặc R2 để chạy kiểm tra này.

## Kịch bản baseline

1. Upload và index bốn PDF một lần trong space benchmark riêng.
2. Nhập dataset, ánh xạ bốn mã và duyệt ground truth.
3. Chạy smoke test một câu đã duyệt, kiểm tra trace/context/citation và adapter Ragas.
4. Chạy đủ 36 câu, mỗi câu 3 lần, history/summary rỗng: **108 mẫu** trên cùng corpus.
5. Xuất JSON/CSV, báo cáo các metric riêng lẻ và theo ba loại câu; kèm số lỗi, số mẫu hợp lệ và tỷ lệ fallback. Không thay ground truth giữa lượt chạy.

Không có kết quả chấm chính thức đi kèm bản nháp. Điểm số chỉ có sau khi người dùng duyệt bộ dữ liệu và cấu hình OpenAI hợp lệ.
