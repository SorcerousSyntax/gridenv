FROM python:3.11-slim
LABEL org.opencontainers.image.title="GridWorld Survival"
LABEL org.opencontainers.image.description="Mini-game RL environment for AI agents"
LABEL org.opencontainers.image.version="1.0.0"

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .
RUN pip install --no-cache-dir -e .

EXPOSE 7860
USER appuser

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]