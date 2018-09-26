build-android:
	expo build:android

build-ios:
	expo build:ios

clean:
	rm -rf node_modules
	rm -rf .expo

install:
	npm install

dev:
	npm start 
