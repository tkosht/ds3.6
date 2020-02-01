ps images down:
	docker-compose $@

im:images

build:
	docker-compose build --no-cache

up:
	docker-compose up -d

active:
	docker-compose up

reup: down up

clean:
	docker-compose down --rmi all
	sudo rm -rf app/__pycache__
