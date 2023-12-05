# Use an official Python runtime as a parent image
FROM docker.io/python:3.11-slim as app

# Set working directory
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update --allow-unauthenticated -y \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt \
    && pip cache purge


# Copy the application files
COPY . .

RUN touch .env
# Set environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Configure Streamlit
RUN mkdir -p /.streamlit \
    && chmod 777 /.streamlit \
    && echo "[general]\nemail = \"\"\n" > /.streamlit/credentials.toml \
    && echo "[server]\nenableCORS = false\nenableXsrfProtection = false\nenableWebsocketCompression = false\n" > /.streamlit/config.toml

# Expose port
EXPOSE 8080

 CMD /bin/bash -c "python backend.py && python -m streamlit run app.py --server.port=8080"
