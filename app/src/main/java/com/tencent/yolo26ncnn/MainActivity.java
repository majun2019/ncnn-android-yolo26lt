// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.yolo26ncnn;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;

import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;

public class MainActivity extends Activity implements SurfaceHolder.Callback
{
    public static final int REQUEST_CAMERA = 100;
    private static final boolean DIAG_OFFLINE_ONLY = false;

    private YOLO26Ncnn yolo26ncnn = new YOLO26Ncnn();
    private int facing = 0;

    private Spinner spinnerTask;
    private TextView textModelName;
    private Spinner spinnerCPUGPU;
    private int current_task = 0;
    private int current_cpugpu = 0;

    private SurfaceView cameraView;
    private SeekBar seekProb;
    private SeekBar seekNms;
    private TextView textProb;
    private TextView textNms;

    private float currentProbThreshold = 0.35f;  // 激进校准后正常检测分数在0.3-0.65
    private float currentNmsThreshold = 0.30f;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        textProb = (TextView) findViewById(R.id.textProb);
        textNms = (TextView) findViewById(R.id.textNms);
        seekProb = (SeekBar) findViewById(R.id.seekProb);
        seekNms = (SeekBar) findViewById(R.id.seekNms);

        seekProb.setProgress((int)(currentProbThreshold * 100));
        seekNms.setProgress((int)(currentNmsThreshold * 100));
        updateThresholdText();

        seekProb.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                int p = Math.max(1, Math.min(99, progress));
                currentProbThreshold = p / 100.0f;
                updateThresholdText();
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                applyDetectThresholds();
            }
        });

        seekNms.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                int p = Math.max(5, Math.min(95, progress));
                currentNmsThreshold = p / 100.0f;
                updateThresholdText();
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                applyDetectThresholds();
            }
        });

        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {

                int new_facing = 1 - facing;

                yolo26ncnn.closeCamera();

                yolo26ncnn.openCamera(new_facing);

                facing = new_facing;
            }
        });

        spinnerTask = (Spinner) findViewById(R.id.spinnerTask);
        spinnerTask.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_task)
                {
                    current_task = position;
                    updateModelNameText();
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        textModelName = (TextView) findViewById(R.id.textModelName);
        updateModelNameText();

        spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_cpugpu)
                {
                    current_cpugpu = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        reload();
    }

    private void reload()
    {
        boolean ret_init = yolo26ncnn.loadModel(getAssets(), current_task, 0, current_cpugpu);
        if (!ret_init)
        {
            Log.e("MainActivity", "yolo26ncnn loadModel failed");
        }

        applyDetectThresholds();
    }

    private static final String[] MODEL_NAMES = {
        "yolo26n_safehat",       // detect
        "yolo26n_seg_e2e",       // segment
        "yolo26n_pose_e2e",      // pose
        "yolo26n_cls",           // classify
        "yolo26n_obb_e2e"        // obb
    };

    private void updateModelNameText()
    {
        if (current_task >= 0 && current_task < MODEL_NAMES.length)
        {
            textModelName.setText(MODEL_NAMES[current_task]);
        }
    }

    private void applyDetectThresholds()
    {
        boolean ok = yolo26ncnn.setDetectThresholds(currentProbThreshold, currentNmsThreshold);
        if (!ok)
        {
            Log.w("MainActivity", "setDetectThresholds failed");
        }
    }

    private void updateThresholdText()
    {
        textProb.setText(String.format("置信度 %.2f", currentProbThreshold));
        textNms.setText(String.format("NMS %.2f", currentNmsThreshold));
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        yolo26ncnn.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (DIAG_OFFLINE_ONLY)
        {
            Log.w("MainActivity", "DIAG_OFFLINE_ONLY enabled: skip camera open");
            return;
        }

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
        {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        yolo26ncnn.openCamera(facing);
    }

    @Override
    public void onPause()
    {
        super.onPause();

        yolo26ncnn.closeCamera();
    }
}
