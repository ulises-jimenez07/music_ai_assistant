#!/bin/bash

# Script to manage the Music AI Assistant services using Docker Compose

# Function to display usage information
show_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start       - Start all services"
    echo "  stop        - Stop all services"
    echo "  restart     - Restart all services"
    echo "  status      - Check the status of all services"
    echo "  logs        - View logs from all services"
    echo "  logs [service] - View logs from a specific service (streamlit-app, llm-service, code-execution-service)"
    echo "  build       - Rebuild all services"
    echo "  down        - Stop and remove all containers, networks"
    echo "  help        - Show this help message"
    echo ""
}

# Check if Docker and Docker Compose are installed
check_dependencies() {
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker compose version &> /dev/null; then
        echo "Error: Docker Compose is not installed or not in PATH"
        exit 1
    fi
}

# Start all services
start_services() {
    echo "Starting Music AI Assistant services..."
    docker compose up -d
    echo "Services started. Access the Streamlit app at http://localhost:8501"
}

# Stop all services
stop_services() {
    echo "Stopping Music AI Assistant services..."
    docker compose stop
    echo "Services stopped."
}

# Restart all services
restart_services() {
    echo "Restarting Music AI Assistant services..."
    docker compose restart
    echo "Services restarted."
}

# Check status of services
check_status() {
    echo "Checking status of Music AI Assistant services..."
    docker compose ps
}

# View logs
view_logs() {
    if [ -z "$1" ]; then
        echo "Showing logs for all services..."
        docker compose logs --tail=100 -f
    else
        echo "Showing logs for $1..."
        docker compose logs --tail=100 -f "$1"
    fi
}

# Rebuild services
build_services() {
    echo "Rebuilding Music AI Assistant services..."
    docker compose build
    echo "Build completed."
}

# Stop and remove containers, networks
down_services() {
    echo "Stopping and removing Music AI Assistant services..."
    docker compose down
    echo "Services removed."
}

# Main script execution
check_dependencies

case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        check_status
        ;;
    logs)
        view_logs "$2"
        ;;
    build)
        build_services
        ;;
    down)
        down_services
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac

exit 0
