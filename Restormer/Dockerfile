FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY . .

RUN conda create -n restormer python=3.7 -y && \
    echo "conda activate restormer" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc && conda activate restormer && \
    conda install -y pytorch=1.8.1 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia && \
    pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm && \
    pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips && \
    pip install streamlit fastapi uvicorn requests python-multipart && \
    apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && \
    python setup.py develop --no_cuda_ext"

WORKDIR /app/Deraining

EXPOSE 8000

# Define the default command to run your application
CMD ["bash", "-c", "source activate restormer && uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000"]