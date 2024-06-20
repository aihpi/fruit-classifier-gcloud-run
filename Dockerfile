# https://fastapi.tiangolo.com/deployment/docker/
# Use the official Python slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /code

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application code
COPY ./app /code/app

# Set environment variables
ENV WANDB_API_KEY=""
ENV WANDB_ORG=""
ENV WANDB_PROJECT=""
ENV WANDB_MODEL_NAME=""
ENV WANDB_MODEL_VERSION=""

# Command to run the application
CMD ["fastapi", "run", "app/main.py", "--port", "8080"]