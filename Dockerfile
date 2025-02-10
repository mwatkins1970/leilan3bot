FROM ghcr.io/huggingface/text-generation-inference:latest

# Install additional requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the handler code
COPY chapter2/handler.py /app/handler.py

# Set environment variables
ENV HUGGINGFACE_HUB_CACHE=/data
ENV TRANSFORMERS_CACHE=/data
ENV HANDLER_PATH=/app/handler.py
