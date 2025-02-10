# Start from Hugging Face's Text Generation Inference base image
FROM ghcr.io/huggingface/text-generation-inference:latest

# Set working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker's caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . . 

# Set environment variables
ENV PYTHONPATH="/app"
ENV HUGGINGFACE_HUB_CACHE=/data
ENV TRANSFORMERS_CACHE=/data
ENV HANDLER_PATH=/app/chapter2/handler.py

# Expose port (if needed)
EXPOSE 80

# Command to run the bot
CMD ["python", "chapter2/main.py", "Leilan"]
