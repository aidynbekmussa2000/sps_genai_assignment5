# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.llm_model import FineTunedGPT2
from app.rnn_model import RNNTextGenerator

app = FastAPI()

# instantiate RNN model (loads model.pt once on startup)
rnn_model = RNNTextGenerator()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int
    

class QuestionRequest(BaseModel):
    question: str
    
llm = FineTunedGPT2()

@app.get("/")
def read_root():
    return {"message": "RNN text generator API is running"}


@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    generated_text = rnn_model.generate_text(
        request.start_word,
        request.length,
    )
    return {"generated_text": generated_text}

@app.post("/answer_with_llm")
def answer_with_llm(request: QuestionRequest):
    answer = llm.answer_question(request.question)
    return {"answer": answer}