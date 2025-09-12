FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# --- Python 3.11 everywhere (matches pip & runtime) ---
RUN ln -sf $(which python3.11) /usr/local/bin/python \
 && ln -sf $(which python3.11) /usr/local/bin/python3

# --- Build deps for llama.cpp + runtime curl/CA ---
RUN apt-get update && apt-get install -y \
    git build-essential cmake curl ca-certificates libcurl4-openssl-dev && \
 rm -rf /var/lib/apt/lists/*

# --- Build llama.cpp from latest master with CUDA and install to /usr/local ---
ARG CUDA_ARCHS="80;86;87,89;90,90a"
RUN git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp && \
    cd /tmp/llama.cpp && \
 export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" && \
 export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:$LIBRARY_PATH" && \
 # Create symlink for libcuda.so.1 if it doesn't exist
 ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
 cmake -B build \
      -DGGML_CUDA=ON \
      -DLLAMA_CURL=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      . && \
 cmake --build build -j $(nproc) && \
 cmake --install build && \
 echo "/usr/local/lib" > /etc/ld.so.conf.d/llama.conf && ldconfig && \
 rm -rf /tmp/llama.cpp

ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"

# --- Python deps ---
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /requirements.txt

# --- App code ---
COPY src/ /src/

# --- Sensible defaults (override at runtime) ---
ENV LLAMA_SERVER_HOST="127.0.0.1" \
    LLAMA_SERVER_PORT="4444" \
    DEFAULT_MAX_TOKENS="32768" \
    B2_S3_ENDPOINT="https://s3.us-east-005.backblazeb2.com" \
    B2_MAX_CONCURRENCY="50" \
    B2_CHUNK_MB="128"

# Start worker
CMD ["python3", "-u", "/src/handler.py"]
