FROM continuumio/miniconda3:24.5.0-0

WORKDIR /workspace

# Install dependencies in an isolated conda environment.
COPY requirements.txt /tmp/requirements.txt
RUN conda create -n plfm -c conda-forge python=3.9 rasterio gdal pip -y && \
    conda run -n plfm pip install --no-cache-dir -r /tmp/requirements.txt && \
    conda clean -afy

# Copy project sources after dependency install to maximize Docker cache reuse.
COPY . /workspace

ENV PYTHONUNBUFFERED=1

# Usage:
# docker run --rm -v ${PWD}:/workspace plfm python preprocess.py ...
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "plfm", "python"]
