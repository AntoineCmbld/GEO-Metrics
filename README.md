# GEO Metrics

## Purpose
GEO Metrics generates analytical reports that evaluate how well content is optimized for generative AI answer engines (Generative Engine Optimization). It ingests content, queries multiple LLM providers, scores visibility/readiness, and produces structured reports.

## Architecture

### Core Services
- **api-gateway** (port 3000): Unified entry point, request routing, and proxy to backend services
- **report-service** (port 8080): Aggregates LLM outputs, computes GEO scores, stores report metadata
- **llm-service** (port 8081): Normalized interface to multiple LLM providers (OpenAI, Groq)
- **db** (port 5432): PostgreSQL database for persistence

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
- [Minikube](https://minikube.sigs.k8s.io/docs/start/) installed
- [kubectl](https://kubernetes.io/docs/tasks/tools/) installed
- [Docker](https://docs.docker.com/get-docker/) installed
- OpenAI and/or Groq API keys

---

### Step 1: Start Minikube

```bash
minikube start
```

Verify Minikube is running:
```bash
minikube status
```

---

### Step 2: Configure Docker for Minikube

Point your Docker CLI to Minikube's Docker daemon so images are built directly in the cluster:

```bash
eval $(minikube docker-env)
```

**Note:** Run this command in each new terminal session where you want to build images.

---

### Step 3: Create Kubernetes Secrets

Create secrets for sensitive data before deploying:

```bash
# Database password
kubectl create secret generic db-secret \
  --from-literal=DB_PASSWORD=your_secure_password

# LLM API keys (replace with your actual keys)
kubectl create secret generic llm-secrets \
  --from-literal=OPENAI_API_KEY=sk-your-openai-key \
  --from-literal=GROQ_API_KEY=gsk_your-groq-key
```

Verify secrets were created:
```bash
kubectl get secrets
```

---

### Step 4: Build Docker Images

Build all service images in Minikube's Docker environment:

```bash
# Build API Gateway
docker build -t api-gateway:latest ./api-gateway

# Build Report Service
docker build -t report-service:latest ./report-service

# Build LLM Service
docker build -t llm-service:latest ./llm-service
```

Verify images are available:
```bash
docker images | grep -E 'api-gateway|report-service|llm-service'
```

---

### Step 5: Deploy to Kubernetes

Apply all Kubernetes manifests:

```bash
kubectl apply -f k8s/
```

This will create:
- 4 Deployments (api-gateway, report-service, llm-service, db)
- 4 Services (ClusterIP for internal communication)
- 1 ConfigMap (database initialization script)

---

### Step 6: Verify Deployment

Check that all pods are running:

```bash
kubectl get pods
```

Expected output:
```
NAME                              READY   STATUS    RESTARTS   AGE
api-gateway-xxxxxxxxxx-xxxxx      1/1     Running   0          30s
report-service-xxxxxxxxxx-xxxxx   1/1     Running   0          30s
llm-service-xxxxxxxxxx-xxxxx      1/1     Running   0          30s
db-xxxxxxxxxx-xxxxx               1/1     Running   0          30s
```

Check services:
```bash
kubectl get services
```

If any pod is not running, check logs:
```bash
kubectl logs -f deployment/api-gateway
kubectl logs -f deployment/report-service
kubectl logs -f deployment/llm-service
kubectl logs -f deployment/db
```

---

### Step 7: Access the API Gateway

#### Option A: Using Minikube Service (Recommended)

```bash
minikube service api-gateway --url
```

This will output a URL like `http://192.168.49.2:30000`. Use this URL to access the API.

#### Option B: Port Forwarding

```bash
kubectl port-forward service/api-gateway 3000:3000
```

Then access via: `http://localhost:3000`

---

### Step 8: Test the API

Test the health endpoint:

```bash
# Replace with your actual URL from Step 7
curl http://localhost:3000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-19T..."
}
```

List available LLM providers:
```bash
curl http://localhost:3000/api/llm/providers
```

Query an LLM provider:
```bash
curl -X POST http://localhost:3000/api/llm/query \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "prompt": "What is GEO?"
  }'
```

Create a report:
```bash
curl -X POST http://localhost:3000/api/reports/ \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Sample content for GEO analysis",
    "llm_responses": [
      {"provider": "openai", "score": 0.85, "response": "..."},
      {"provider": "groq", "score": 0.78, "response": "..."}
    ]
  }'
```

---

## Environment Variables

### API Gateway
- `PORT`: Server port (default: 3000)
- `LLM_SERVICE_URL`: LLM service endpoint
- `REPORT_SERVICE_URL`: Report service endpoint

### Report Service
- `PORT`: Server port (default: 8080)
- `DB_HOST`: Database host
- `DB_PORT`: Database port (default: 5432)
- `DB_NAME`: Database name
- `DB_USER`: Database user
- `DB_PASSWORD`: Database password (from secret)

### LLM Service
- `PORT`: Server port (default: 8081)
- `OPENAI_API_KEY`: OpenAI API key (from secret)
- `GROQ_API_KEY`: Groq API key (from secret)

### Database
- `POSTGRES_DB`: Database name
- `POSTGRES_USER`: Database user
- `POSTGRES_PASSWORD`: Database password (from secret)