# Build stage
FROM ghcr.io/huggingface/text-generation-inference:latest AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y make wget && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir text-generation==0.7.0 && \
    make install-server || true

# Final stage
FROM ghcr.io/huggingface/text-generation-inference:latest

# Copy built artifacts from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages


# Install additional Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy your Python code
COPY app /app
COPY chapter2 /chapter2
COPY ems /ems

# Create /embeddings folder & download them from HF
RUN mkdir -p /embeddings /embeddings/subchunked && \
    wget -O /embeddings/dialogue_embeddings_mpnet.npy https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/dialogue_embeddings_mpnet.npy && \
    wget -O /embeddings/essay_embeddings_mpnet.npy https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/essay_embeddings_mpnet.npy && \
    wget -O /embeddings/interview_embeddings_mpnet.npy https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/interview_embeddings_mpnet.npy && \
    wget -O /embeddings/dialogue_chunks_mpnet.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/dialogue_chunks_mpnet.json && \
    wget -O /embeddings/dialogue_metadata_mpnet.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/dialogue_metadata_mpnet.json && \
    wget -O /embeddings/essay_chunks_mpnet.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/essay_chunks_mpnet.json && \
    wget -O /embeddings/essay_metadata_mpnet.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/essay_metadata_mpnet.json && \
    wget -O /embeddings/interview_chunks_mpnet.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/interview_chunks_mpnet.json && \
    wget -O /embeddings/interview_intros_mpnet.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/interview_intros_mpnet.json && \
    wget -O /embeddings/interview_metadata_mpnet.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/interview_metadata_mpnet.json && \
    wget -O /embeddings/subchunked/dialogue_metadata_subchunked.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/subchunked/dialogue_metadata_subchunked.json && \
    wget -O /embeddings/subchunked/dialogue_texts_subchunked.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/subchunked/dialogue_texts_subchunked.json && \
    wget -O /embeddings/subchunked/essay_chunks_mpnet.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/subchunked/essay_chunks_mpnet.json && \
    wget -O /embeddings/subchunked/essay_metadata_mpnet.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/subchunked/essay_metadata_mpnet.json && \
    wget -O /embeddings/subchunked/essay_stats_mpnet.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/subchunked/essay_stats_mpnet.json && \
    wget -O /embeddings/subchunked/interview_chunks_mpnet.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/subchunked/interview_chunks_mpnet.json && \
    wget -O /embeddings/subchunked/interview_metadata_mpnet.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/subchunked/interview_metadata_mpnet.json && \
    wget -O /embeddings/subchunked/interview_stats_mpnet.json https://huggingface.co/datasets/mwatkins1970/leilan3-embeddings/resolve/main/subchunked/interview_stats_mpnet.json

# Set handler path for HF Endpoints
ENV HANDLER_PATH=/app/handler.py
EXPOSE 80
CMD ["bash", "-c", "text-generation-launcher & exec python /chapter2/main.py Leilan"]

