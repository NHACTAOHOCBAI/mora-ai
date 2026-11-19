"""Ragas structured-output boundary with one retry policy: the OpenAI SDK."""
from ragas.llms.base import InstructorBaseRagasLLM
from app.core.usage_tracking import record_usage

JUDGE_REASON_INSTRUCTIONS = (
    "Hãy viết mọi trường reason, kể cả trong các mục lồng nhau, bằng tiếng Việt "
    "bất kể ngôn ngữ của câu hỏi, tài liệu hay ví dụ. Giữ nguyên thuật ngữ hoặc "
    "trích dẫn gốc khi cần. Áp dụng tiêu chí và cấu trúc đầu ra theo yêu cầu."
)


class ResponsesJudge(InstructorBaseRagasLLM):
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def generate(self, prompt, response_model):
        raise TypeError("Benchmark judge requires async agenerate()")

    async def agenerate(self, prompt, response_model):
        response = await self.client.responses.parse(
            model=self.model, input=prompt, instructions=JUDGE_REASON_INSTRUCTIONS,
            text_format=response_model,
            store=False, max_output_tokens=8192
        )
        if response.status != "completed" or response.output_parsed is None:
            raise ValueError("Judge response incomplete or refused")
        record_usage(response, self.model, "benchmark_judge")
        return response.output_parsed
