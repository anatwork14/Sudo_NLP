FROM python:3.12-slim

ENV PYTHONUNBUFFERED 1
ENV APP_HOME /usr/src/app

WORKDIR $APP_HOME

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY setup_files.py .
COPY app.py .
COPY model.pkl .
COPY tokenizer.pkl .

# Expose the port the application runs on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]