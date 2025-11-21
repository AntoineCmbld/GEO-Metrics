# GEO Metrics

## Purpose
GEO Metrics generates analytical reports that evaluate how well content is optimized for generative AI answer engines (Generative Engine Optimization). It ingests content, queries multiple LLM models, scores visibility/readiness, and produces structured reports.

## Architecture

### Core Services
- **api-gateway** (port 3000): Unified entry point, request orchestration
- **report-service** (port 8080): Aggregates LLM outputs, computes GEO scores, stores reports
- **llm-service** (port 8081): Interface to HuggingFace models
- **db** (port 5432): PostgreSQL database for report persistence

### Service Communication
```
Client → API Gateway (3000)
           ↓
           ├→ Report Service (8080) → Database (5432)
           └→ LLM Service (8081)
```

---

## Kubernetes Deployment

### Prerequisites
- [Minikube](https://minikube.sigs.k8s.io/docs/start/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Docker](https://docs.docker.com/get-docker/)
- HuggingFace API key

---

### Setup

```bash
./setup.sh
```

The script will prompt for your database password and HuggingFace API key, then handle all setup automatically.

---

## API Endpoints

After setup, use the displayed API URL:

### Health Check
```bash
curl $API_URL/api/health
```

### List Available Models
```bash
curl $API_URL/api/llm/models
```

### Full GEO Analysis
```bash
curl -X POST $API_URL/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "brand_name": "Tesla",
    "models": ["meta-llama/Llama-3.1-8B-Instruct"],
    "keywords": ["electric vehicles", "sustainable energy"],
    "prompt_template": "What do you know about {keyword}? What brands come to your mind when you think of {keyword}?"
  }'
```

### Query LLM Directly
```bash
curl -X POST $API_URL/api/llm/query \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "What is Generative Engine Optimization?"
  }'
```

### List All Reports
```bash
curl $API_URL/api/reports
```

### Get Specific Report
```bash
curl $API_URL/api/reports/1
```

---

## Available Models

- `openai/gpt-oss-20b`
- `meta-llama/Llama-3.1-8B-Instruct`
- `zai-org/GLM-4.6`

---

## Development

### Rebuild After Code Changes

```bash
./rebuild.sh all                 # All services
./rebuild.sh api-gateway         # Specific service
./rebuild.sh llm-service
./rebuild.sh report-service
```

---

## Cleanup

```bash
./teardown.sh
```