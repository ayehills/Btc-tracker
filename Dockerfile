FROM python:3.11-slim

WORKDIR /app

# scipy / scikit-learn ship manylinux wheels, so no compiler toolchain needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

# Production WSGI server (not `flask run`). Shell form so $PORT expands. One
# worker fits the VM; two threads keep the live price endpoint responsive while
# a model fit is in progress.
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 2 --timeout 120
