"""Capture the inputs/outputs used by Ragas without issuing extra provider calls."""
import numpy as np
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.embeddings import OpenAIEmbeddings

DETAILS_VERSION = 1


class RecordingJudge(InstructorBaseRagasLLM):
    def __init__(self, inner):
        self.inner = inner
        self.outputs = []

    def generate(self, prompt, response_model):
        raise TypeError("Use async agenerate()")

    async def agenerate(self, prompt, response_model):
        result = await self.inner.agenerate(prompt, response_model)
        self.outputs.append({"schema": response_model.__name__, "data": result.model_dump()})
        return result


class RecordingEmbeddings(OpenAIEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.original = None
        self.generated = []

    async def aembed_text(self, text, **kwargs):
        vector = await super().aembed_text(text, **kwargs)
        self.original = vector
        return vector

    async def aembed_texts(self, texts, **kwargs):
        vectors = await super().aembed_texts(texts, **kwargs)
        self.generated = vectors
        return vectors


def details_for(name, outputs, embeddings, contexts, complete):
    details = {"version": DETAILS_VERSION, "complete": complete}
    def records(schema):
        return [item["data"] for item in outputs if item["schema"] == schema]
    if name == "contextPrecision":
        rows = []
        cumulative = 0
        for index, result in enumerate(records("ContextPrecisionOutput")):
            verdict = result["verdict"]
            cumulative += verdict
            precision = cumulative / (index + 1)
            context = contexts[index] if index < len(contexts) else {}
            rows.append({"rank": index + 1, "context": context, "verdict": verdict,
                         "reason": result["reason"], "cumulativeRelevant": cumulative,
                         "precisionAtK": precision, "contribution": precision * verdict})
        details.update(rows=rows, expectedContexts=len(contexts), relevantCount=cumulative,
                       numerator=sum(row["contribution"] for row in rows),
                       epsilon=1e-10, denominator=cumulative + 1e-10)
    elif name == "faithfulness":
        generated = records("StatementGeneratorOutput")
        nli = records("NLIStatementOutput")
        statements = nli[-1]["statements"] if nli else []
        details.update(generatedStatements=generated[-1]["statements"] if generated else [],
                       statements=statements, supportedCount=sum(bool(s["verdict"]) for s in statements),
                       statementCount=len(statements))
    elif name == "answerRelevance":
        rows = []
        used_index = 0
        original = np.asarray(embeddings.original) if embeddings.original is not None else None
        original_norm = float(np.linalg.norm(original)) if original is not None else None
        for result in records("AnswerRelevanceOutput"):
            used = bool(result["question"])
            row = {**result, "used": used, "dotProduct": None, "generatedNorm": None, "cosine": None}
            if used:
                if original is not None and used_index < len(embeddings.generated):
                    vector = np.asarray(embeddings.generated[used_index])
                    if vector.shape == original.shape and np.isfinite(original).all() and np.isfinite(vector).all():
                        dot = float(np.dot(original, vector))
                        norm = float(np.linalg.norm(vector))
                        denominator = original_norm * norm
                        row.update(dotProduct=dot, generatedNorm=norm,
                                   cosine=dot / denominator if denominator else None)
                used_index += 1
            rows.append(row)
        used_rows = [row for row in rows if row["used"]]
        similarities = [row["cosine"] for row in used_rows if row["cosine"] is not None]
        factor = int(not all(row["noncommittal"] for row in used_rows)) if used_rows else 0
        details.update(rows=rows, originalNorm=original_norm if original_norm is not None and np.isfinite(original_norm) else None, usedCount=len(used_rows),
                       meanCosine=sum(similarities) / len(similarities) if similarities else None,
                       substantiveFactor=factor)
    # Provider vectors live only in memory, never in persisted details or exports.
    return details
