FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && apt-get clean

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /code

# project metadata + README
COPY pyproject.toml README.md /code/

# trained LSTM model + fine-tuned GPT-2
COPY model.pt /code/model.pt
COPY models/gpt2_squad /code/models/gpt2_squad

# install deps
RUN uv sync

# app code
COPY ./app /code/app

CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]