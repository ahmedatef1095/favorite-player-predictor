# Step 1: Use an official Python runtime as a parent image
FROM python:3.10-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file into the container
COPY requirements.txt .

# Step 4: Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your application files into the container
COPY app.py .
COPY players_dataset.csv .
COPY templates/ ./templates/
COPY mlruns/ ./mlruns/

# Step 6: Make the port the app runs on available to the outside world
EXPOSE 5001

# Step 7: Define the command to run your app
# We use --host=0.0.0.0 to make it accessible from outside the container
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]