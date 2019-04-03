package com.example.jerico.cprassistant;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.video.FarnebackOpticalFlow;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Window;
import android.view.WindowManager;
import android.view.SurfaceView;
import android.widget.Button;
import android.widget.TextView;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String  TAG              = "MainActivity";

    private static final double SCALE = 0.1;

    private Mat mRgba;
    private Mat prevFrameBGR;

    private boolean isWorking;

    private CameraBridgeViewBase mOpenCvCameraView;
    private FarnebackOpticalFlow opticalFlow;
    private Button mStartButton;
    private Button mMoreButton;
    private TextView mCCRTextView;



    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
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

        isWorking = false;

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        mOpenCvCameraView.enableFpsMeter();
        mOpenCvCameraView.setCameraIndex(1);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);


        mCCRTextView = (TextView) findViewById(R.id.ccr_textview);
        mMoreButton = (Button) findViewById(R.id.more_button);
        mStartButton = (Button) findViewById(R.id.start_button);
        mStartButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                isWorking = !isWorking;
                if (isWorking) {
                    mMoreButton.setVisibility(View.GONE);
                    mStartButton.setText("Stop");
                    mCCRTextView.setText("DETECTING");
                } else {
                    mMoreButton.setVisibility(View.VISIBLE);
                    mStartButton.setText("Start");
                    mCCRTextView.setText("READY");
                }
            }
        });
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
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
        prevFrameBGR = new Mat();
        opticalFlow = FarnebackOpticalFlow.create(1, 0.5, true, 10, 1, 5, 1.2, 0);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();

        if (isWorking) {
            Size scaleSize = new Size(mRgba.size().width * SCALE,mRgba.size().height * SCALE);
            Mat currentFrameBGR = new Mat();
            Imgproc.resize(mRgba, currentFrameBGR, scaleSize);
            Imgproc.blur(currentFrameBGR, currentFrameBGR, new Size(1, 1));

            if (!prevFrameBGR.empty()) {
                Mat prevFrameGray = new Mat();
                Mat currentFrameGray = new Mat();
                Imgproc.cvtColor(prevFrameBGR, prevFrameGray, Imgproc.COLOR_RGB2GRAY);
                Imgproc.cvtColor(currentFrameBGR, currentFrameGray, Imgproc.COLOR_RGB2GRAY);

                Mat flow = new Mat();
                opticalFlow.calc(prevFrameGray, currentFrameGray, flow);

                List<Mat> channels = new ArrayList<Mat>();
                Mat magnitude = new Mat();
                Mat direction = new Mat();
                Core.split(flow, channels);
                Core.cartToPolar(channels.get(0), channels.get(1), magnitude, direction, true);



            }

            prevFrameBGR = currentFrameBGR;
        }
        return mRgba;
    }
}