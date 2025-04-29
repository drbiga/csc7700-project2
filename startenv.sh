# IMPORTANT!
# Run each command separately
# This script is not to be run using the command line

docker run --name es01 --net elastic -p 9200:9200 -d -m 1GB docker.elastic.co/elasticsearch/elasticsearch:9.0.0
docker exec -it es01 /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic
docker exec -it es01 /usr/share/elasticsearch/bin/elasticsearch-create-enrollment-token -s kibana
export ELASTIC_PASSWORD="3AQp0+4gk+pQubOM_8AG"
eyJ2ZXIiOiI4LjE0LjAiLCJhZHIiOlsiMTcyLjI1LjAuMjo5MjAwIl0sImZnciI6IjMwYWMyNDg2Yzk4OGMxOTk0NzVkZTJlNTdiZjAxOGRlYWRhOWIwZjRmMjczMmIyMDRjNjRhODg4YjAzNTRiOTUiLCJrZXkiOiI0bEo3ZjVZQnA4VHNPV0ZtRTRjdTpGb0t3UExhdElrZmdkODFoMERTeU9nIn0=
docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt .
curl --cacert http_ca.crt -u elastic:$ELASTIC_PASSWORD https://localhost:9200
docker run --name kib01 -d --net elastic -p 5601:5601 docker.elastic.co/kibana/kibana:9.0.0