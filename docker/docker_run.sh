if [ ! "$(docker ps -aq -f name=proof)" ]; then
	docker build ./ -t proof
fi
if [ "$(docker ps -aq -f status=exited -f name=proof)" ]; then
	docker container rm proof
fi
docker run -it -d --name proof -v "$(pwd)":/proof-sharing proof
docker attach proof
