#!/bin/bash

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  GEO-Metrics Teardown Script${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Delete Kubernetes resources
echo -e "${YELLOW}Deleting Kubernetes resources...${NC}"
kubectl delete -f k8s/ --ignore-not-found=true
echo -e "${GREEN}✅ Kubernetes resources deleted${NC}"
echo ""

# Delete secrets
echo -e "${YELLOW}Deleting secrets...${NC}"
kubectl delete secret db-secret --ignore-not-found=true
kubectl delete secret llm-secrets --ignore-not-found=true
echo -e "${GREEN}✅ Secrets deleted${NC}"
echo ""

# Ask if user wants to stop minikube
read -p "Stop minikube? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Stopping minikube...${NC}"
    minikube stop
    echo -e "${GREEN}✅ Minikube stopped${NC}"
fi
echo ""

# Ask if user wants to delete minikube cluster
read -p "Delete minikube cluster? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Deleting minikube cluster...${NC}"
    minikube delete
    echo -e "${GREEN}✅ Minikube cluster deleted${NC}"
fi
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Teardown Complete! 🧹${NC}"
echo -e "${GREEN}========================================${NC}"