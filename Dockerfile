FROM python:3.12.3-alpine3.20

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

ENTRYPOINT ["python"]

CMD ["chatbot.py"]

