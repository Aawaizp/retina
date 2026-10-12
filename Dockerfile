# RETINA local app in a container:
#   docker build -t retina .
#   docker run -p 127.0.0.1:8000:8000 -v retina-runs:/app/data/live retina
#   (thinking runs with an LLM: add  -e GROQ_API_KEY  - keys are passed when the container starts,
#    never stored in the image)
# Then open http://localhost:8000/docs . Every row it makes is SIMULATED.
# The image holds the code, the store profiles, the processed public data layer and (when trained)
# the student model - NOT the raw Instacart files or the per-shopper profiles (Instacart terms).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
WORKDIR /app

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY retina ./retina
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

COPY stores ./stores
COPY scenarios ./scenarios
COPY data/data_sources.json ./data/data_sources.json
COPY data/processed ./data/processed
COPY models ./models

RUN useradd --create-home retina && mkdir -p data/live && chown -R retina /app
USER retina
VOLUME /app/data/live
EXPOSE 8000
CMD ["retina", "app", "--host", "0.0.0.0", "--port", "8000"]
