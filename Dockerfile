FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime libs used by Pillow/Matplotlib in manylinux wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -c "from pathlib import Path; p=Path('requirements.txt'); b=p.read_bytes(); \
txt=b.decode('utf-8') if b'\x00' not in b else b.decode('utf-16'); \
p.write_text(txt, encoding='utf-8', newline='\n')" \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

EXPOSE 5000

# Use Flask CLI so app.py __main__ debug run is not used.
CMD ["python", "-m", "flask", "--app", "app", "run", "--host=0.0.0.0", "--port=5000"]
