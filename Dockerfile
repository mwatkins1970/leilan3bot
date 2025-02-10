# Use the Hugging Face TGI (Text Generation Inference) image
FROM ghcr.io/huggingface/text-generation-inference:latest

# [Optional] Install text-generation server if you want a specific version
# (If you need EXACT version or custom steps; otherwise skip or adjust.)
RUN apt-get update && apt-get install -y make && \
    pip install --no-cache-dir text-generation==0.7.0 && \
    make install-server || true

# Install additional Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy your code
COPY app /app
COPY chapter2 /chapter2
COPY ems /ems

# Let HF Endpoints/TGI know where your custom handler is located
ENV HANDLER_PATH=/app/handler.py

# Expose port 80 (what TGI uses by default)
EXPOSE 80

# Finally, run text-generation-launcher
CMD ["text-generation-launcher"]
    