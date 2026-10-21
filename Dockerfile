# MediReach web app. Build: docker build -t medireach .   Run: docker run -p 8000:8000 medireach
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MEDIREACH_HOST=0.0.0.0 \
    PORT=8000

WORKDIR /srv/medireach
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the running app reads: code, SQL, frontend, and the built database, model and artifacts.
COPY medireach/ medireach/
COPY sql/ sql/
COPY web/ web/
COPY models/ models/
COPY artifacts/ artifacts/
COPY data/medireach.db data/medireach.db

# Run as a normal user; it only writes the demo workflow database (data/ops.db).
RUN useradd --create-home medireach && chown -R medireach data
USER medireach

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/api/health', timeout=4)"
CMD ["python", "-m", "medireach.web"]
