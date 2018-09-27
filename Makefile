build-android:
	make lint
	expo build:android

build-ios:
	make lint
	expo build:ios

clean:
	rm -rf node_modules
	rm -rf .expo

install:
	npm install

dev:
	npm start 

lint:
	grunt eslint
