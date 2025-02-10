# Use the Hugging Face TGI (Text Generation Inference) image
FROM ghcr.io/huggingface/text-generation-inference:latest

# Install only what's absolutely necessary
RUN pip install --no-cache-dir text-generation==0.7.0

# Install additional Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir requests

# Copy your Python code
COPY app /app
COPY chapter2 /chapter2
COPY ems /ems

# Create necessary directories
RUN mkdir -p /embeddings /embeddings/subchunked && \
    mkdir -p /tmp/text-generation-server

# Set handler path for HF Endpoints
ENV HANDLER_PATH=/app/handler.py
EXPOSE 80

# Make sure text-generation-server paths are in PATH
ENV PATH="/usr/local/bin:/tmp/text-generation-server:${PATH}"

# Verify PATH and directory setup
RUN echo "Checking PATH setup:" && \
    echo $PATH && \
    echo "Checking directory existence:" && \
    ls -la /usr/local/bin && \
    ls -la /tmp/text-generation-server || true

CMD ["bash", "-c", "text-generation-launcher & exec python /chapter2/main.py Leilan"]