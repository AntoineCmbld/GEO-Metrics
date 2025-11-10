# GEO Metrics API

**GEO Metrics** is a FastAPI backend that analyzes how frequently a brand appears when discussing specific keywords using the most popular AI models available to the public. The API is containerized with Docker and can be run via Docker Compose.

---

## Features

- Accepts a brand name and a list of keywords
- Queries multiple popular AIs for each keyword
- Counts brand mentions and calculates normalized mention ratio
- Returns results in structured JSON
- Fully containerized with Docker for easy deployment
- Handles OpenAI API errors gracefully per keyword

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/)
- OpenAI, Google Gemini and Groq API key

---

## Setup

1. Clone the repository:

```bash
git clone <repository_url>
cd GEO\ Metrics
```

2. Create a .env file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxx
```

3. Build and start the Docker Compose services:

```bash
docker-compose up --build
```

The API will be available at http://localhost:8000.

Get API documentation at http://localhost:8000/docs#/.