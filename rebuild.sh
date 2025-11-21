#!/bin/bash

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Rebuilding Services${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Configure Docker for minikube
echo -e "${YELLOW}Configuring Docker for minikube...${NC}"
eval $(minikube docker-env)
echo ""

# Determine which service to rebuild
if [ "$1" == "all" ] || [ -z "$1" ]; then
    echo -e "${YELLOW}Rebuilding all services...${NC}"
    docker build -t api-gateway:latest ./api-gateway
    docker build -t report-service:latest ./report-service
    docker build -t llm-service:latest ./llm-service
    
    kubectl rollout restart deployment/api-gateway
    kubectl rollout restart deployment/report-service
    kubectl rollout restart deployment/llm-service
    
    kubectl wait --for=condition=ready pod -l app=api-gateway --timeout=60s
    kubectl wait --for=condition=ready pod -l app=report-service --timeout=60s
    kubectl wait --for=condition=ready pod -l app=llm-service --timeout=60s
    
elif [ "$1" == "api-gateway" ] || [ "$1" == "gateway" ]; then
    echo -e "${YELLOW}Rebuilding api-gateway...${NC}"
    docker build -t api-gateway:latest ./api-gateway
    kubectl rollout restart deployment/api-gateway
    kubectl wait --for=condition=ready pod -l app=api-gateway --timeout=60s
    
elif [ "$1" == "report-service" ] || [ "$1" == "reports" ]; then
    echo -e "${YELLOW}Rebuilding report-service...${NC}"
    docker build -t report-service:latest ./report-service
    kubectl rollout restart deployment/report-service
    kubectl wait --for=condition=ready pod -l app=report-service --timeout=60s
    
elif [ "$1" == "llm-service" ] || [ "$1" == "llm" ]; then
    echo -e "${YELLOW}Rebuilding llm-service...${NC}"
    docker build -t llm-service:latest ./llm-service
    kubectl rollout restart deployment/llm-service
    kubectl wait --for=condition=ready pod -l app=llm-service --timeout=60s
    
else
    echo "Usage: ./rebuild.sh [all|api-gateway|report-service|llm-service]"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Rebuild complete!${NC}"

# Get API URL
MINIKUBE_IP=$(minikube ip)
API_URL="http://$MINIKUBE_IP:30000"
echo -e "${GREEN}API URL:${NC} $API_URL"