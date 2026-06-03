
package com.tencent.yolo26ncnn;

import android.content.res.AssetManager;
import android.view.Surface;

public class YOLO26Ncnn
{
    public native boolean loadModel(AssetManager mgr, int taskid, int modelid, int cpugpu);
    public native boolean setDetectThresholds(float probThreshold, float nmsThreshold);
    public native boolean openCamera(int facing);
    public native boolean closeCamera();
    public native boolean setOutputWindow(Surface surface);

    static {
        System.loadLibrary("yolo26ncnn");
    }
}
