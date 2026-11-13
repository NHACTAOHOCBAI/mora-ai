from pydantic import BaseModel, Field
from typing import List

class StudyCitation(BaseModel):
    documentId: int
    documentName: str
    pageNumber: int = Field(ge=1)
    quote: str

class GeneratedFlashcard(BaseModel):
    front: str
    back: str
    citation: StudyCitation

class GeneratedQuizQuestion(BaseModel):
    question: str
    options: List[str] = Field(min_length=4, max_length=4)
    correctOptionIndex: int = Field(ge=0, le=3)
    explanation: str
    citation: StudyCitation

class StudyGenerateRequest(BaseModel):
    spaceId: int
    documentIds: List[int]
    cardCount: int = Field(ge=5, le=15)
    context: List['StudyContextItem'] = []

class StudyContextItem(BaseModel):
    documentId: int
    documentName: str
    pageNumber: int = Field(ge=1)
    text: str

class StudyGenerateResponse(BaseModel):
    title: str
    flashcards: List[GeneratedFlashcard]
    quizQuestions: List[GeneratedQuizQuestion]
