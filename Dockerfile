FROM nvcr.io/nvidia/tritonserver:26.03-py3

RUN pip install --no-cache-dir \
    opencv-python-headless \
    pillow