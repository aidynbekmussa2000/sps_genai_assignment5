## How to run

1. Clone and cd:
   git clone https://github.com/aidynbekmussa2000/sps_genai_assignment5.git
   cd sps_genai_assignment5

2. Install dependencies:
   uv sync

3. Fine-tune GPT-2 (creates models/gpt2_squad/):
   uv run python fine_tune_gpt2.py

4. Build and run Docker:
   docker build -t sps-genai .
   docker run -p 8000:80 sps-genai

5. Open http://127.0.0.1:8000/docs and use POST /answer_with_llm

6. If Docker is not available, you can run the API directly with
uv run fastapi dev app/main.py

