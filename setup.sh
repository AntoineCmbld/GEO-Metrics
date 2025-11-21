#!/bin/bash

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  GEO-Metrics Setup Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v minikube &> /dev/null; then
    echo -e "${RED}❌ minikube not found. Please install: https://minikube.sigs.k8s.io/docs/start/${NC}"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl not found. Please install: https://kubernetes.io/docs/tasks/tools/${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ docker not found. Please install: https://docs.docker.com/get-docker/${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites found${NC}"
echo ""

# Prompt for secrets
echo -e "${YELLOW}Please enter your credentials:${NC}"
read -sp "Database password: " DB_PASSWORD
echo ""
read -sp "HuggingFace API key: " HUGGINGFACE_API_KEY
echo ""
echo ""

if [ -z "$DB_PASSWORD" ] || [ -z "$HUGGINGFACE_API_KEY" ]; then
    echo -e "${RED}❌ Credentials cannot be empty${NC}"
    exit 1
fi

# Start minikube
echo -e "${YELLOW}Starting minikube...${NC}"
if minikube status &> /dev/null; then
    echo -e "${GREEN}✅ Minikube already running${NC}"
else
    minikube start
    echo -e "${GREEN}✅ Minikube started${NC}"
fi
echo ""

# Configure Docker for minikube
echo -e "${YELLOW}Configuring Docker for minikube...${NC}"
eval $(minikube docker-env)
echo -e "${GREEN}✅ Docker configured${NC}"
echo ""

# Pull required images into minikube
echo -e "${YELLOW}Pulling required Docker images...${NC}"
docker pull postgres:13
docker pull adminer:latest
echo -e "${GREEN}✅ Images pulled${NC}"
echo ""

# Delete existing secrets if they exist
echo -e "${YELLOW}Cleaning up old secrets...${NC}"
kubectl delete secret db-secret --ignore-not-found=true
kubectl delete secret llm-secrets --ignore-not-found=true
echo -e "${GREEN}✅ Old secrets cleaned${NC}"
echo ""

# Create secrets
echo -e "${YELLOW}Creating Kubernetes secrets...${NC}"
kubectl create secret generic db-secret \
  --from-literal=DB_PASSWORD="$DB_PASSWORD"

kubectl create secret generic llm-secrets \
  --from-literal=HUGGINGFACE_API_KEY="$HUGGINGFACE_API_KEY"

echo -e "${GREEN}✅ Secrets created${NC}"
echo ""

# Build Docker images
echo -e "${YELLOW}Building Docker images...${NC}"
echo "Building api-gateway..."
docker build -t api-gateway:latest ./api-gateway

echo "Building report-service..."
docker build -t report-service:latest ./report-service

echo "Building llm-service..."
docker build -t llm-service:latest ./llm-service

echo "Building frontend..."
docker build -t frontend:latest ./frontend

echo -e "${GREEN}✅ Docker images built${NC}"
echo ""

# Deploy to Kubernetes
echo -e "${YELLOW}Deploying to Kubernetes...${NC}"
kubectl apply -f k8s/

echo -e "${GREEN}✅ Deployed to Kubernetes${NC}"
echo ""

# Wait for pods to be ready
echo -e "${YELLOW}Waiting for pods to be ready (this may take a minute)...${NC}"
kubectl wait --for=condition=ready pod -l app=db --timeout=120s
kubectl wait --for=condition=ready pod -l app=llm-service --timeout=120s
kubectl wait --for=condition=ready pod -l app=report-service --timeout=120s
kubectl wait --for=condition=ready pod -l app=api-gateway --timeout=120s
kubectl wait --for=condition=ready pod -l app=frontend --timeout=120s
kubectl wait --for=condition=ready pod -l app=adminer --timeout=120s

echo -e "${GREEN}✅ All pods ready${NC}"
echo ""

# Get URLs
MINIKUBE_IP=$(minikube ip)
FRONTEND_URL="http://$MINIKUBE_IP:30002"
API_URL="http://$MINIKUBE_IP:30000"
ADMINER_URL="http://$MINIKUBE_IP:30001"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete! 🚀${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}Frontend URL:${NC} $FRONTEND_URL"
echo -e "${GREEN}API URL:${NC}      $API_URL"
echo -e "${GREEN}Adminer URL:${NC}  $ADMINER_URL"
echo ""
echo -e "${YELLOW}Test endpoints:${NC}"
echo "  Health check:     curl $API_URL/api/health"
echo "  List models:      curl $API_URL/api/llm/models"
echo "  Full analysis:    curl -X POST $API_URL/api/analyze \\"
echo "                      -H \"Content-Type: application/json\" \\"
echo "                      -d '{"
echo "                        \"brand_name\": \"Tesla\","
echo "                        \"models\": [\"meta-llama/Llama-3.1-8B-Instruct\"],"
echo "                        \"keywords\": [\"electric vehicles\"]"
echo "                      }'"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "  View pods:        kubectl get pods"
echo "  View services:    kubectl get services"
echo "  View logs:        kubectl logs -l app=<service-name>"
echo "  API docs:         $API_URL/docs"
echo ""
echo -e "${YELLOW}To tear down:${NC}"
echo "  ./teardown.sh"
echo ""