# Use the Hugging Face TGI (Text Generation Inference) image
FROM ghcr.io/huggingface/text-generation-inference:latest

# Install dependencies with proper error checking
RUN apt-get update && \
    apt-get install -y make wget python3 python3-pip && \
    pip install --no-cache-dir text-generation==0.7.0 && \
    make install-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir requests

# Copy your Python code
COPY app /app
COPY chapter2 /chapter2
COPY ems /ems

# Create embeddings directory (but don't download - that happens at runtime)
RUN mkdir -p /embeddings /embeddings/subchunked

# Set handler path for HF Endpoints
ENV HANDLER_PATH=/app/handler.py
EXPOSE 80

# Make sure text-generation-server is in PATH
ENV PATH="/usr/local/bin:${PATH}"

CMD ["bash", "-c", "text-generation-launcher & exec python /chapter2/main.py Leilan"]