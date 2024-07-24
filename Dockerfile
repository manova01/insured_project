# Use a Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the application files to the container
COPY app.py /app/
COPY ridge_model.pkl /app/
COPY vectorizer.pkl /app/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
