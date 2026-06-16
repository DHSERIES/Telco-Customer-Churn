FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system deps needed for some packages (numpy/pandas/xgboost may need build tools)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . /app

# expose default FastAPI port
EXPOSE 8000

# Default command runs the FastAPI app. Override CMD to run streamlit if desired:
# docker run -p 8501:8501 <image> streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
