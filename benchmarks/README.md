# Benchmark RAG v1

## Giải thích điểm từng câu

Các run mới lưu `metricDetailsVersion=1` trong cấu hình đóng băng. Mỗi metric
trong `metricsJson` có thêm `details`; điểm, trạng thái, bảng tổng hợp và CSV
giữ nguyên ý nghĩa. JSON export chứa cả chi tiết. Không cần migration database.

Mở một câu/lần lặp ở trang admin để xem:

- Context Precision: thứ tự context, nhãn hữu ích và lý do judge, tổng tích lũy,
  precision ở từng vị trí và từng đóng góp vào Average Precision. Mẫu số dùng
  epsilon `1e-10` đúng như Ragas.
- Faithfulness: phát biểu đã tách, phán quyết được hỗ trợ và lý do; tỷ lệ dùng
  đúng số phát biểu được kiểm tra trong đầu ra NLI của Ragas.
- Answer Relevance: câu hỏi tạo ngược, cờ mơ hồ, câu rỗng bị bỏ, tích vô hướng,
  độ dài hai vector, cosine từng cặp, trung bình và hệ số mơ hồ. Vector đầy đủ
  chỉ tồn tại trong bộ nhớ, không lưu hoặc xuất ra. Không cắt cosine âm về 0.
- Citation Accuracy: nguồn chuẩn và các cặp tài liệu–trang đã loại trùng,
  cặp đúng/sai và tử số/mẫu số. Document ID không xác định vẫn tính sai.
- Latency: các thời gian thành phần và chênh lệch còn lại so với tổng đo được;
  không bao gồm thời gian judge. Thời gian chấm mỗi metric được hiển thị riêng.

Các dữ liệu này được ghi từ chính các lời gọi Ragas hiện có, không gọi AI thêm
để tạo lời giải thích. Lý do judge giữ nguyên ngôn ngữ đầu ra. Khi lỗi/timeout,
chỉ giữ phần dữ liệu đã quan sát và đánh dấu phép tính chưa hoàn tất.
Run cũ không bị chấm lại; giao diện giữ điểm gốc và thông báo thiếu chi tiết.
Các số hiển thị được làm tròn, điểm vẫn dùng giá trị đầy đủ do Ragas trả về.

Benchmark đo pipeline đang dùng trong chat, gồm Router, Hybrid (dense + BM25), RRF,
FlashRank, Synthesis, Evaluator và các nhánh GENERAL/fallback. Không ép router vào
RAG và không đưa ground truth vào bước sinh câu trả lời.

## Chuẩn bị

1. Cài dependencies bằng `venv\Scripts\python.exe -m pip install -r requirements.txt`.
2. Cấu hình toàn bộ OpenAI cho Benchmark trên trang Admin: key judge riêng, model judge và
   model embedding judge. Backend không cần biến môi trường Benchmark riêng; AI service đọc
   cấu hình qua API nội bộ có xác thực.
3. Khởi động PostgreSQL, Qdrant, Backend, AI và Frontend như README các repo.
   Hibernate `ddl-auto=update` tạo bốn bảng benchmark mới; không sửa bảng chat.
4. Đăng nhập admin, mở `/admin/benchmarks`, nhập
   `benchmarks/datasets/operating-systems-v1.json`.
5. Tải bốn PDF tương ứng từ `../mora-data`. Mã tài liệu, SHA-256 và số trang được
   kiểm tra trước khi dùng lại luồng R2 → parse → index. Chờ tất cả ở trạng thái READY.
   Mỗi dataset tạo một space riêng. PDF lỗi cần kiểm tra key/quota/R2, sau đó nhập
   dataset mới để chuẩn bị lại; v1 không thay corpus của dataset đã tải.
6. Đọc PDF gốc theo liên kết trang, sửa/duyệt từng ground truth. Mọi sửa đổi câu hỏi
   hủy duyệt câu đó. Ground truth nháp không phải kết quả được xác nhận tự động.
7. Chạy smoke test (câu đầu, một lần) và kiểm tra trace. Sau khi smoke thành công,
   chạy chính thức: 36 câu × 3 lần = 108 mẫu. Mỗi câu có history/summary rỗng và được
   tìm trong toàn bộ bốn PDF, không lọc theo tài liệu ground truth.

Không tự chạy benchmark trả phí trong quá trình cài đặt. Cấu hình model mặc định có
thể đổi bằng biến môi trường; dùng model mà tài khoản OpenAI của bạn được phép gọi.
Model embedding retrieval hiện tại lấy từ cấu hình Admin, phải khớp với
corpus đã index. Judge embedding độc lập với embedding retrieval.
`RETRIEVAL_ENABLED` được ghi trong snapshot: khi tắt, benchmark đo nhánh context thô
giống chat hiện tại. Muốn đo Hybrid/RRF/FlashRank, hoàn tất reindex theo README chính
rồi bật `RETRIEVAL_ENABLED=true` trước khi tạo run; không đổi cấu hình giữa run.

## Chỉ số và cách đọc

- **Context Precision**: Ragas `ContextPrecision`, dùng đáp án chuẩn và danh sách
  context theo đúng thứ tự đưa vào Synthesis; không context thì 0.
- **Faithfulness**: Ragas `Faithfulness`, so câu trả lời với context đã dùng.
  Không context thì `not_applicable`. Khi RAG đã thử rồi fallback, trace vẫn giữ
  context đã dùng ở lần thử đó để đánh giá câu trả lời cuối cùng so với nguồn.
- **Answer Relevance**: Ragas `AnswerRelevancy`, OpenAI tạo ba câu hỏi ngược và
  OpenAI embeddings đo liên quan. Không dùng điểm Evaluator nội bộ làm điểm benchmark.
- **Citation Accuracy**: số cặp `(mã tài liệu, trang PDF)` khớp ground truth chia
  tổng số cặp trích dẫn duy nhất. Không citation là 0; sai/thiếu document ID tính sai.
  Đây là độ chính xác tài liệu–trang, không đo tính đúng của từng quote hay recall.
- **Latency**: `perf_counter` từ lúc bắt đầu pipeline AI đến lúc tạo xong response,
  gồm thời gian router, retrieval, mọi lần synthesis và Evaluator, cả fallback.
  Loại trừ indexing, chuẩn bị corpus, HTTP Backend, hàng đợi và chấm Ragas.

Mean tính trên các mẫu có giá trị, không thay lỗi bằng 0 hoặc 1. Mỗi metric công bố
riêng số mẫu hợp lệ, lỗi, thiếu và không áp dụng. Latency thêm median/p95 nội suy
tuyến tính; hiển thị tỷ lệ fallback, lỗi và rerank thực tế. Không có điểm tổng hoặc
ngưỡng đạt/rớt. So sánh điểm giữa các nhóm cần đọc cả số mẫu bị lỗi/thiếu.

Ragas được khóa ở 0.4.3; adapter dùng collections API, OpenAI Responses adapter và
OpenAIEmbeddings, được kiểm thử offline với biên OpenAI mock. Tài liệu tham khảo:
[metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/).

## Lưu trữ và phục hồi

Backend lưu dataset, câu hỏi, run và từng sample trong PostgreSQL. Run đóng băng
câu hỏi, ground truth, nguồn, document IDs, raw context, cấu hình model/retrieval,
phiên bản thư viện và commit của ba repo. Không lưu key trong snapshot hoặc export.
Corpus Qdrant được kiểm tra fingerprint; BM25 được phục hồi từ chunk hiện có sau
restart. Khi tạo run mới, tài liệu thiếu chunk OpenAI được lập lại chỉ mục từ các
trang đã lưu do Backend gửi; không tải/đọc lại PDF và không dùng ground truth.
Chỉ tài liệu thiếu chỉ mục mới được tạo embedding, có thể phát sinh phí OpenAI.
Khi thực thi/tiếp tục run đã đóng băng, hệ thống không tự sửa corpus; thiếu chunk
hoặc fingerprint thay đổi sẽ báo lỗi để tránh trộn dữ liệu trong cùng một run.

Một Backend instance chạy một run tại một thời điểm, các sample tuần tự. Không
triển khai nhiều replica worker trong v1. Khi restart, RUNNING/PREPARING được đánh
dấu INTERRUPTED. Tiếp tục dùng snapshot cũ và không chạy lại sample đã hoàn tất.
Sample đang sinh lúc bị ngắt có thể cần gọi lại; provider không hỗ trợ exactly-once.
Nút thử lại judge chỉ chấm lại JUDGE_ERROR từ trace đã lưu, không sinh lại answer.
Không tiếp tục khi cấu hình/commit thay đổi; tạo run mới thay vì trộn cấu hình.
Không chỉnh/xóa space hoặc tài liệu nguồn trong khi chạy benchmark.

Trace ghi cả lỗi provider bị pipeline bắt để không tính thông báo lỗi thành câu trả
lời hợp lệ. Rerank unavailable/failure vẫn là fallback hợp lệ của pipeline, được ghi
trạng thái thực tế; không giả định mọi mẫu đều đã rerank.

Evaluator thử lại tối đa hai lần với lỗi tạm thời 429/500/502/503/504, chờ 2 giây rồi
4 giây; toàn bộ thời gian chờ được tính vào latency. Nếu vẫn lỗi hoặc phản hồi không
đúng schema, QC có trạng thái `error`, không gán điểm 1.0 và không báo QC passed.
Câu trả lời đã sinh được giữ trong trace để kiểm tra, nhưng sample benchmark được
đánh dấu GENERATION_ERROR và không chấm như một mẫu thành công. Không sinh lại câu
trả lời hoặc chuyển GENERAL chỉ vì API Evaluator tạm thời không hoạt động.

## API

Frontend gọi Backend với JWT admin tại `/api/admin/benchmarks`:

- `GET/POST /datasets`, `GET /datasets/{id}`, `GET /datasets/{id}/export`.
- `PUT /datasets/{id}/questions/{questionId}` và
  `POST /datasets/{id}/questions/{questionId}/approval` (`approved: boolean`).
- `POST /datasets/{id}/documents/{code}` multipart `file`.
- `POST /datasets/{id}/runs` (`smoke: boolean`), `GET /runs`, `GET /runs/{id}`.
- `POST /runs/{id}/resume`, `POST /runs/{id}/retry-judge`.
- `GET /runs/{id}/export?format=json|csv`.

Backend gọi AI tại `/api/internal/benchmarks/parse`, `/index`, `/config`, `/prepare`, `/execute`,
`/score` không yêu cầu token nội bộ. Các API này dành cho mạng local/private giữa
hai service; API quản trị tại Backend vẫn yêu cầu JWT và quyền admin.

## Kiểm tra offline

```powershell
.\venv\Scripts\python.exe benchmarks\validate_dataset.py
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Backend: `mvnw.cmd "-Dtest=Benchmark*Test,UserServiceImplTest" test`.
Frontend: `npm.cmd run build`.

Chạy thật với OpenAI/R2/Qdrant và duyệt ground truth là bước riêng. Kết quả mock
trong test không phải kết quả benchmark của đề tài.

Judge gọi Responses API trực tiếp qua adapter Ragas, chỉ OpenAI SDK thực hiện retry.
`BENCHMARK_JUDGE_TIMEOUT=45` giây mỗi request và `BENCHMARK_JUDGE_MAX_RETRIES=1`
(tối đa hai lần gửi); không có retry Instructor bọc bên ngoài. Mỗi metric có
gân sách tổng 180 giây, lỗi được lưu riêng thay vì giả định điểm đạt.
Thay đổi adapter/retry làm thay đổi cấu hình đóng băng; tạo run mới để dùng cấu hình mới.
