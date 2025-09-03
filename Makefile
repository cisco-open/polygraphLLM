.PHONY: help install spacy server frontend dev clean

# Default target
help:
	@echo "PolygraphLLM - Simple Make Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  install    Install package and dependencies"
	@echo "  server     Start backend server"
	@echo "  frontend   Start frontend development server"
	@echo "  dev        Start both backend and frontend"
	@echo "  clean      Clean build artifacts"

# Installation
install:
	pip install -e .
	pip install -r requirements.txt
	python3 -m spacy download en_core_web_sm

# Start backend server
server:
	python3 server.py

# Start frontend
frontend:
	cd playground && npm i && npm run dev

# Start both backend and frontend
dev:
	@echo "Starting backend and frontend..."
	@trap 'kill %1; kill %2' INT; \
	make server & \
	make frontend & \
	wait

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
