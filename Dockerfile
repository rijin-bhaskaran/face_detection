# Slim, multi-arch base (works on x86_64 and ARM edge devices such as Raspberry Pi).
FROM python:3.12-slim

# Minimal system libraries required by opencv-python-headless (OpenMP + GLib).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so they're cached independently of source changes.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app.py config.json ./
COPY cascades ./cascades
COPY models ./models
RUN mkdir -p videos

# No display inside a container; mount a video file or pass through a camera device at run time.
ENTRYPOINT ["python", "app.py"]
CMD ["--no-display"]
