"""Ragas structured-output boundary with one retry policy: the OpenAI SDK."""
from ragas.llms.base import InstructorBaseRagasLLM


class ResponsesJudge(InstructorBaseRagasLLM):
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def generate(self, prompt, response_model):
        raise TypeError("Benchmark judge requires async agenerate()")

    async def agenerate(self, prompt, response_model):
        response = await self.client.responses.parse(
            model=self.model, input=prompt, text_format=response_model,
            store=False, max_output_tokens=8192
        )
        if response.status != "completed" or response.output_parsed is None:
            raise ValueError("Judge response incomplete or refused")
        return response.output_parsed
