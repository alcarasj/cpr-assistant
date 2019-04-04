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
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Window;
import android.view.WindowManager;
import android.view.SurfaceView;
import android.widget.Button;
import android.widget.TextView;
import android.view.View;

import java.util.ArrayList;
import java.util.List;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Timer;
import java.util.TimerTask;
import java.lang.Runnable;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String  TAG              = "MainActivity";

    private static final double SCALE = 0.025;
    private static final double MIN_FLOW_THRESHOLD = 0.3;
    private static final double AVERAGING_FRAMES = 10;
    private static final int MINIMUM_ACCELERATION = 150;
    private static final int MIN_UPWARD_ACCEL_TIME_MS = 750;

    private Mat mRgba;
    private Mat prevFrameBGR;

    private boolean isWorking;
    private boolean isBreathing;
    private boolean downwardAccelDetected;
    private boolean upwardAccelDetected;

    private CameraBridgeViewBase mOpenCvCameraView;
    private FarnebackOpticalFlow opticalFlow;
    private Button mStartButton;
    private Button mMoreButton;
    private TextView mCCRTextView;

    private Queue<int[]> buffer;
    private Timer detectionTimer;
    private int detectedCompressions;
    private double lastDetectedCompressionTime;
    private int verticalAcceleration;
    private int ccr;



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
                    isBreathing = false;
                    detectedCompressions = 0;
                    lastDetectedCompressionTime = 0;
                    ccr = 0;
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
        detectedCompressions = 0;
        ccr = 0;
        lastDetectedCompressionTime = 0;
        downwardAccelDetected = false;
        upwardAccelDetected = false;
        buffer = new LinkedList<int[]>();
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        prevFrameBGR = new Mat();
        opticalFlow = FarnebackOpticalFlow.create(3, 0.5, true, 8, 1, 3, 1.1, 0);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        int width = mRgba.cols();
        int height = mRgba.rows();


        if (isWorking) {
            Size scaleSize = new Size(width * SCALE,height * SCALE);
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

                Imgproc.threshold(magnitude, magnitude, MIN_FLOW_THRESHOLD, 0, Imgproc.THRESH_TOZERO);

                Mat upwardMovementMask = directionalMaskCreator(direction, "UP");
                Mat upwardMovement = new Mat();
                Core.bitwise_and(magnitude, magnitude, upwardMovement, upwardMovementMask);
                int upwardSum = (int) Core.sumElems(upwardMovement).val[0];

                Mat downwardMovementMask = directionalMaskCreator(direction, "DOWN");
                Mat downwardMovement = new Mat();
                Core.bitwise_and(magnitude, magnitude, downwardMovement, downwardMovementMask);
                int downwardSum = (int) Core.sumElems(downwardMovement).val[0];

                Mat lateralMovementMask = directionalMaskCreator(direction, "LATERAL");
                Mat lateralMovement = new Mat();
                Core.bitwise_and(magnitude, magnitude, lateralMovement, lateralMovementMask);
                int lateralSum = (int) Core.sumElems(lateralMovement).val[0];

                int totalPixels = width * height;
                Mat magnitudeBinary = new Mat();
                Imgproc.threshold(magnitude, magnitudeBinary, 0, 1, Imgproc.THRESH_BINARY);
                int pixelsMoved = (int) Core.sumElems(magnitudeBinary).val[0];
                int totalMovementPCG = (int) (pixelsMoved / totalPixels);

                int verticalDisplacement = upwardSum - downwardSum;
                int verticalDisplacementAvg = 0;
                int verticalVelocity = 0;
                verticalAcceleration = 0;
                int data[] = new int[3];

                if (buffer.size() >= AVERAGING_FRAMES) {
                    int[] headData = buffer.remove();
                    int verticalDisplacementSum = 0;
                    for (int[] item: buffer) {
                        verticalDisplacementSum += item[0];
                    }
                    verticalDisplacementAvg = (int) (verticalDisplacementSum / AVERAGING_FRAMES);
                    int prevDisplacementAvg = headData[2];

                    verticalVelocity = verticalDisplacementAvg - prevDisplacementAvg;
                    int prevVelocity = headData[1];

                    verticalAcceleration = verticalVelocity - prevVelocity;
                }

                data[0] = verticalDisplacement;
                data[1] = verticalVelocity;
                data[2] = verticalDisplacementAvg;
                buffer.add(data);

                if (!downwardAccelDetected && verticalAcceleration < -(MINIMUM_ACCELERATION * 0.3)) {
                    downwardAccelDetected = true;
                    TimerTask resetTask = new TimerTask() {
                        @Override
                        public void run() {
                           downwardAccelDetected = false;
                           detectionTimer.cancel();
                        }
                    };
                    detectionTimer = new Timer();
                    detectionTimer.schedule(resetTask, MIN_UPWARD_ACCEL_TIME_MS);
                } else if (downwardAccelDetected && verticalAcceleration > MINIMUM_ACCELERATION) {
                    upwardAccelDetected = true;
                }

                if (downwardAccelDetected && upwardAccelDetected) {
                    detectedCompressions++;
                    upwardAccelDetected = false;
                    downwardAccelDetected = false;
                    long currentTime = System.currentTimeMillis();
                    if (detectedCompressions > 0) {
                        double timeDiff = currentTime - lastDetectedCompressionTime;
                        ccr = (int) (60 / (timeDiff / 1000 ));
                        Log.e("LOL", "" + lastDetectedCompressionTime);
                     }
                    lastDetectedCompressionTime = currentTime;
                }
                Handler refresh = new Handler(Looper.getMainLooper());
                refresh.post(new Runnable() {
                    public void run() {
                        mCCRTextView.setText("N: " + detectedCompressions + ", CCR: " + ccr + ", ACC: " + verticalAcceleration);
                    }
                });

            }

            prevFrameBGR = currentFrameBGR;
        }
        return mRgba;
    }

    public Mat directionalMaskCreator(Mat matrix, String mode) {
        Mat mask = Mat.zeros(matrix.rows(), matrix.cols(), CvType.CV_8UC1);
        for (int i = 0; i < mask.rows(); i++) {
            for (int j = 0; j < mask.cols(); j++) {
                double[] vector = matrix.get(i, j);
                boolean condition = false;
                int value = (int) vector[0];
                switch (mode){
                    case "UP":
                        condition = value > 45 && value < 135;
                        break;
                    case "DOWN":
                        condition = value > 225 && value < 315;
                        break;
                    case "LATERAL":
                        condition = (value >= 160 && value <= 200) || (value <= 20 || value >= 340);
                        break;
                    default:
                        break;
                }

                if (condition) {
                    mask.put(i, j, 1);
                }
                else {
                    mask.put(i, j, 0);
                }
            }
        }
        return mask;
    }
}