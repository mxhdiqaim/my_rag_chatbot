# RAG Chatbot with Swaemauri SDK, Groq, Flask, and Docker

This project is a fully functional, containerized **Retrieval-Augmented Generation (RAG)** chatbot. It uses the high-speed Groq API for language model inference and is served via a Flask web server. It is challenge for Swarmauri Job Application.

The code in this repo show how to build a **RAG** system that answers questions based on a predefined knowledge base, complete with session management to handle conversation history.

## Features

- **Retrieval-Augmented Generation (RAG):** The chatbot uses a local vector store to find relevant context before generating an answer, making its responses accurate and grounded in the provided data.
- **High-Speed Inference:** Powered by Swarmauri SDK, Groq API with the Llama 3 model for near-instant responses.
- **Web API:** A robust Flask API allows interaction with the chatbot over HTTP (The Flask setup is very basic as I am new to python web applications)
- **Dockerized:** Fully containerized with Docker and Docker Compose for easy, consistent setup and deployment.
- **Production-Ready:** Uses Gunicorn as the WSGI server for stable production deployments.

## Tech Stack

- **Backend:** Python, Flask
- **LLM:** Groq (Llama 3)
- **RAG & Vector Store:** LangChain, Scikit-learn, Swarmauri SDK
- **Containerization:** Docker, Docker Compose
- **Deployment:** Gunicorn

## Getting Started

Follow these instructions to get the project running on your local machine.

### Prerequisites

- Git
- Docker and Docker Compose
- A Groq API Key (get one from [groq.com](https://groq.com/))

### Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone <github-repo-url>
    cd my_rag_chatbot
    ```

2.  **Create the environment file:**
    Create a file named `.env` in the root of the project. This file will hold your secret API key.
    ```bash
    GROQ_API_KEY=your_groq_api_key_here
    ```

### Running the Application Locally

This project uses Docker Compose to build the container and run the application. The `app.py` (LangChain version) is the default entry point.

1.  **Build and run the container:**

    ```bash
    docker-compose up --build
    ```

    This command will build the Docker image based on the `Dockerfile` and start the service.

2.  The server will be running at `http://localhost:5000`. You will see logs from Gunicorn in your terminal.

## Interacting with the API

You can send `POST` requests to the `/chat` endpoint using any API client like `curl` or Postman.

The API expects a JSON body with a `query`.

**Example with `curl`:**

1.  **First question:**

    ```bash
    curl -X POST http://localhost:5000/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "What is the highest mountain?"}'
    ```

2.  **Follow-up question:**
    ```bash
    curl -X POST http://localhost:5000/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "Where is it located?"}'
    ```

**Or through the deployed link:**

1.  **First question:**

    ```bash
    curl -X POST https://my-rag-chatbot-8wk0.onrender.com/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "What is the highest mountain?"}'
    ```

2.  **Follow-up question:**
    ```bash
    curl -X POST https://my-rag-chatbot-8wk0.onrender.com/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "Where is it located?"}'
    ```

## Project Structure

```bash
.
├── app.py                  # Main web server (LangChain implementation)
├── swarmauri_chatbot.py    # Alternative implementation without web server
├── chatbot.py              # Original command-line chatbot (LangChain)
├── Dockerfile              # Instructions to build the Docker image
├── docker-compose.yml
```
