ps:
	docker-compose ps

im:images

images:
	docker-compose images

build:
	docker-compose build --no-cache

up:
	docker-compose up -d

active:
	docker-compose up

down:
	docker-compose down

reup: down up

clean:
	docker-compose down --rmi all
	sudo rm -rf app/__pycache__
