# GEO Metrics

## Purpose
GEO Metrics generates analytical reports that evaluate how well content is optimized for generative AI answer engines (Generative Engine Optimization). It ingests content, queries multiple LLM providers, scores visibility/readiness, and produces structured reports.

## Core Services
- api-gateway: Unified entry point, request routing, auth.
- report-service: Aggregates raw LLM outputs, computes GEO scores, stores report metadata.
- llm-service: Normalized interface to multiple LLM providers (OpenAI, Gemini, Groq).
- db: Persistence (reports, provider responses, scoring history).
- k8s: Deployment manifests.

### Prerequisites
- [Minikube](https://minikube.sigs.k8s.io/docs/start/) installed and running
- [kubectl](https://kubernetes.io/docs/tasks/tools/) configured
- [Helm](https://helm.sh/docs/intro/install/) (optional, if using charts)
- Docker (for building images locally, if needed)

---

### 1. Start Minikube
```bash
minikube start