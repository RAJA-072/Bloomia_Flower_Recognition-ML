# Use an appropriate base image with Python 3.8 or later
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Command to run your application
CMD ["python", "app.py"]  # Replace with your actual entry point
