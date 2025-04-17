docker build -t code-execution-service:latest -f code_execution_service/Dockerfile .
docker run -d -p 8082:8082 --name code-execution-service code-execution-service:latest
docker stop code-execution-service
docker rm code-execution-service

docker build -t code-generation-service:latest -f llm_service/Dockerfile .
docker run -p 8081:8081 --name code-generation-service --add-host host.docker.internal:host-gateway code-generation-service:latest
docker stop code-generation-service
docker rm code-generation-services
