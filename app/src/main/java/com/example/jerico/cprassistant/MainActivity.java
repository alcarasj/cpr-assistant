package com.example.jerico.cprassistant;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.video.FarnebackOpticalFlow;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.view.Window;
import android.view.WindowManager;
import android.view.SurfaceView;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String  TAG              = "MainActivity";

    private boolean mhasProgramStarted = false;
    private Button startButton;
    private Mat mRgba;
    private Mat mPrevGray;
    private Mat mFlow;
    private Mat mGray;
    private Mat mMagnitude;
    private Mat mDirection;
    private List<Mat> mChannels;
    private FarnebackOpticalFlow mFOFlow;

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully!");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        startButton = (Button) findViewById(R.id.start_button);
        startButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mhasProgramStarted = !mhasProgramStarted;
                if (mhasProgramStarted) {
                    startButton.setText("Stop");
                } else {
                    startButton.setText("Start");
                }
            }
        });

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        mOpenCvCameraView.enableFpsMeter();
        mOpenCvCameraView.setCameraIndex(1);
        mOpenCvCameraView.setMaxFrameSize(360, 270);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mPrevGray = new Mat(height, width, CvType.CV_8UC1);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        mFlow = new Mat(height, width, CvType.CV_8UC2);
        mMagnitude = new Mat(height, width, CvType.CV_8UC1);
        mDirection = new Mat(height, width, CvType.CV_8UC1);
        mChannels = new ArrayList<Mat>();
        mFOFlow = FarnebackOpticalFlow.create(1, 0.5, true, 10, 1, 5, 1.2, 0);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (!mhasProgramStarted) {
            mPrevGray = mGray;
        } else {
            try {
                mFOFlow.calc(mPrevGray, mGray, mFlow);
                Core.split(mFlow, mChannels);
                Core.cartToPolar(mChannels.get(0), mChannels.get(1), mMagnitude, mDirection, true);
                Log.e("SUCCESS", "MAG: " + Core.mean(mMagnitude));
                Log.e("SUCCESS", "DIR: " + Core.mean(mDirection));
            } catch (Exception e) {
                Log.e("THROW_ERROR", e.getMessage());
            }
        }

        mPrevGray = mGray;

        return mRgba;
    }
}