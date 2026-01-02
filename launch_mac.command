#!/bin/bash
cd "$(dirname "$0")"

echo "Checking for Docker..."
if ! command -v docker &> /dev/null
then
    echo "Error: Docker is not installed or not running."
    echo "Please install Docker Desktop from https://www.docker.com/products/docker-desktop/"
    read -p "Press [Enter] to exit..."
    exit 1
fi

echo "Launching BALLS! (v18)..."
echo "Rebuilding container to ensure latest fixes..."
docker-compose down
docker-compose build --no-cache
docker-compose up

read -p "Press [Enter] to close..."