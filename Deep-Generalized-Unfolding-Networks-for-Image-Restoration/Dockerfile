FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY . .

RUN conda create -n dgu python=3.7 -y && \
    echo "conda activate dgu" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc && conda activate dgu && \
    conda install -y pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia && \
    pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm && \
    pip install streamlit fastapi uvicorn requests python-multipart && \
    cd pytorch-gradual-warmup-lr && python setup.py install && cd .. && \
    apt-get update && apt-get install ffmpeg libsm6 libxext6  -y"

WORKDIR /app/Deraining

EXPOSE 8000

# Define the default command to run your application
CMD ["bash", "-c", "source activate dgu && uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000"]