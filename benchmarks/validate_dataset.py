"""Offline checks for a benchmark draft against the original PDF bytes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader


def validate(dataset_path: Path, pdf_directory: Path) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    documents = dataset["documents"]
    questions = dataset["questions"]
    assert len(documents) == 4, "Expected four PDF documents"
    assert len({d["code"] for d in documents}) == 4, "Duplicate document codes"
    counts = Counter(q["type"] for q in questions)
    assert counts == {"DIRECT": 12, "SYNTHESIS": 12, "VISUAL": 12}, counts
    assert len({q["code"] for q in questions}) == 36, "Duplicate question codes"
    assert len({q["question"] for q in questions}) == 36, "Duplicate questions"
    by_code = {d["code"]: d for d in documents}
    for document in documents:
        pdf = pdf_directory / document["fileName"]
        assert pdf.resolve().parent == pdf_directory.resolve(), "Invalid PDF filename"
        assert hashlib.sha256(pdf.read_bytes()).hexdigest() == document["sha256"], pdf
        assert len(PdfReader(pdf).pages) == document["pageCount"], pdf
    for question in questions:
        for field in ("question", "referenceAnswer", "explanation"):
            assert question[field].strip(), (question["code"], field)
        assert isinstance(question["approved"], bool), question["code"]
        assert question["sources"], question["code"]
        for source in question["sources"]:
            document = by_code[source["documentCode"]]
            assert 1 <= source["pageNumber"] <= document["pageCount"], (question["code"], source)
    return {"documents": len(documents), "questions": len(questions), "types": dict(counts),
            "approved": sum(q["approved"] for q in questions), "pdf_hashes": "verified"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "datasets" / "operating-systems-v1.json")
    parser.add_argument("--pdf-directory", type=Path, default=Path(__file__).resolve().parents[2] / "mora-data")
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset, args.pdf_directory), ensure_ascii=False, indent=2))
