# Use an official lightweight Python image
FROM python:3.12.5-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (including libGL)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variables
ENV CONFIG_PATH="/app/config.json"

# Expose the Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]


#docker run -d -p 7860:7860 gradio-app
#docker buildx build --platform=linux/amd64 -t floralarea:x86 .
#docker tag floralarea:x86 eai6/floralarea:x86 
#docker push eai6/floralarea:x86