#!/bin/bash

# BryceGPT Deployment Script for Google Cloud Run
# This script automates the deployment process

set -e  # Exit on error

echo "🚀 BryceGPT Deployment Script"
echo "================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: No Google Cloud project configured"
    echo "Please run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "📋 Project ID: $PROJECT_ID"

# Configuration
SERVICE_NAME="brycegpt"
REGION="us-central1"
MEMORY="2Gi"
CPU="2"
TIMEOUT="300"

echo ""
echo "⚙️  Configuration:"
echo "   Service Name: $SERVICE_NAME"
echo "   Region: $REGION"
echo "   Memory: $MEMORY"
echo "   CPU: $CPU"
echo "   Timeout: ${TIMEOUT}s"
echo ""

# Confirm deployment
read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled"
    exit 0
fi

# Build the container
echo ""
echo "🔨 Building container image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
echo ""
echo "🚢 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory $MEMORY \
  --cpu $CPU \
  --timeout $TIMEOUT

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo "================================"
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "📚 API Endpoints:"
echo "   Health Check: $SERVICE_URL/health"
echo "   Generate: $SERVICE_URL/generate"
echo "   Vocabulary: $SERVICE_URL/vocab"
echo "   Docs: $SERVICE_URL/docs"
echo ""
echo "💡 Test your deployment:"
echo "   curl $SERVICE_URL/health"
echo ""
echo "🔧 Update your frontend with this URL:"
echo "   API_URL = \"$SERVICE_URL\""

