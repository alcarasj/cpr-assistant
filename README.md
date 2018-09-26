# CPR Assistant (Dissertation, Master in Computer Science)
## Installation
This project requires Node to be installed on your machine.
1. Install dependencies with `npm install`.
2. Start the Expo server with `npm start`.

## Grunt
Grunt CLI must be installed on your machine for linting with ESLint.
1. Install Grunt CLI with `npm install -g grunt-cli`.
2. Run the ESLint task with `grunt eslint`.

## Builds
Expo CLI must be installed on your machine for building APK/IPA's. You must also be a registered user on Expo.
1. Run `make build-ios` or `make build-android` to initiate an Expo build for iOS or Android, respectively.
2. The terminal will show a URL to the initiated build job on Expo - go to this URL to see the progress and download the completed APK/IPA build.
