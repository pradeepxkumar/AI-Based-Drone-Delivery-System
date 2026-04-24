FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for geopandas/shapely/GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY app.py .
COPY src/ ./src/
COPY data/ ./data/
COPY output/ ./output/
COPY .streamlit/ ./.streamlit/

# Create a non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user

# Expose port 7860 (HF Spaces expects this)
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.fileWatcherType=none"]
