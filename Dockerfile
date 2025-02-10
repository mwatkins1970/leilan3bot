# Use the Hugging Face TGI (Text Generation Inference) image
FROM ghcr.io/huggingface/text-generation-inference:latest

# Install missing text-generation-server
RUN apt-get update && apt-get install -y make && \
    pip install --no-cache-dir text-generation==0.7.0 && \
    make install-server || true  # Avoid failure if make install-server is not needed

# Install additional Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full application code
COPY chapter2/ chapter2/
COPY ems/ ems/

# Set necessary environment variables
ENV HUGGINGFACE_HUB_CACHE=/data
ENV TRANSFORMERS_CACHE=/data
ENV HANDLER_PATH=/app/handler.py

# Expose the correct port
EXPOSE 80

# Command to start the text generation server
CMD ["text-generation-launcher"]
