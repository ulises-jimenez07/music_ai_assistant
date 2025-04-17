docker build -t code-execution-service:latest -f code_execution_service/Dockerfile .
docker run -p 8082:8082 --name code-execution-service code-execution-service:latest
docker start code-execution-service
docker stop code-execution-service
docker rm code-execution-service

docker build -t code-generation-service:latest -f llm_service/Dockerfile .
docker run -p 8081:8081 --name code-generation-service --add-host host.docker.internal:host-gateway code-generation-service:latest
docker start code-generation-service
docker stop code-generation-service
docker rm code-generation-services


docker build -t music-app:latest -f ./Dockerfile .
docker run -p 8501:8501 --name music-app music-app:latest
