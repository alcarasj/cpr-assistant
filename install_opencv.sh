wget https://sourceforge.net/projects/opencvlibrary/files/opencv-ios/4.0.1/opencv-4.0.1-ios-framework.zip
unzip -a opencv-3.4.1-ios-framework.zip
cd ios
cp -r ./../opencv2.framework ./
cd ..
rm -rf opencv-3.4.1-ios-framework.zip
rm -rf opencv2.framework/

wget https://sourceforge.net/projects/opencvlibrary/files/opencv-android/4.0.1/opencv-4.0.1-android-sdk.zip
unzip opencv-3.4.1-android-sdk.zip
cd android/app/src/main
mkdir jniLibs
cp -r ./../../../../OpenCV-android-sdk/sdk/native/libs/ ./jniLibs
cd ../../../../
rm -rf opencv-3.4.1-android-sdk.zip
rm -rf OpenCV-android-sdk/
