// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2026 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolo26.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static YOLO26* g_yolo26 = 0;
static ncnn::Mutex lock;
static const bool kConversionOfflineProbeEnabled = false;
static int g_current_taskid = -1;  // 0=det,1=seg,2=pose,3=cls,4=obb  updated in loadModel

static const char* taskid_to_name(int taskid)
{
    static const char* names[] = {"det", "seg", "pose", "cls", "obb"};
    if (taskid >= 0 && taskid <= 4) return names[taskid];
    return "unknown";
}

static bool asset_exists(AAssetManager* mgr, const char* path)
{
    if (!mgr || !path)
        return false;

    AAsset* asset = AAssetManager_open(mgr, path, AASSET_MODE_STREAMING);
    if (!asset)
        return false;

    AAsset_close(asset);
    return true;
}

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    // yolo26
    {
        ncnn::MutexLockGuard g(lock);

        if (g_yolo26)
        {
            std::vector<Object> objects;

            // --- YOLO26BENCH latency instrumentation ---
            static int   s_last_taskid = -1;
            static int   s_frame_count = 0;
            static float s_ms_sum      = 0.f;
            static float s_ms_min      = 1e9f;
            static float s_ms_max      = 0.f;

            if (g_current_taskid != s_last_taskid)
            {
                // Task switched: reset per-task counters
                s_last_taskid = g_current_taskid;
                s_frame_count = 0;
                s_ms_sum      = 0.f;
                s_ms_min      = 1e9f;
                s_ms_max      = 0.f;
                __android_log_print(ANDROID_LOG_INFO, "YOLO26BENCH",
                    "TASK_SWITCH task=%d name=%s",
                    g_current_taskid, taskid_to_name(g_current_taskid));
            }

            double t0 = ncnn::get_current_time();
            g_yolo26->detect(rgb, objects);
            double t1 = ncnn::get_current_time();
            float detect_ms = (float)(t1 - t0);

            s_ms_sum += detect_ms;
            if (detect_ms < s_ms_min) s_ms_min = detect_ms;
            if (detect_ms > s_ms_max) s_ms_max = detect_ms;
            s_frame_count++;

            // Per-frame log (every frame) — parseable by collect_latency.py
            __android_log_print(ANDROID_LOG_INFO, "YOLO26BENCH",
                "FRAME task=%d name=%s detect_ms=%.2f objects=%zu",
                g_current_taskid, taskid_to_name(g_current_taskid),
                detect_ms, objects.size());

            // Running summary every 30 frames
            if (s_frame_count % 30 == 0)
            {
                __android_log_print(ANDROID_LOG_INFO, "YOLO26BENCH",
                    "SUMMARY task=%d name=%s frames=%d mean_ms=%.2f min_ms=%.2f max_ms=%.2f fps_mean=%.1f",
                    g_current_taskid, taskid_to_name(g_current_taskid),
                    s_frame_count,
                    s_ms_sum / s_frame_count,
                    s_ms_min, s_ms_max,
                    1000.f / (s_ms_sum / s_frame_count));
            }
            // ------------------------------------------

            g_yolo26->draw(rgb, objects);
        }
        else
        {
            draw_unsupported(rgb);
        }
    }

    draw_fps(rgb);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_yolo26;
        g_yolo26 = 0;
    }

    ncnn::destroy_gpu_instance();

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int taskid, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolo26ncnn_YOLO26Ncnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint taskid, jint modelid, jint cpugpu)
{
    // taskid: 0=det, 1=seg, 2=pose, 3=cls, 4=obb
    // modelid: 保留但不使用（只有yolo26n E2E模式）
    // cpugpu: 0=CPU, 1=GPU, 2=turnip
    if (taskid < 0 || taskid > 4 || cpugpu < 0 || cpugpu > 2)
    {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    // 任务后缀
    const char* tasknames[5] =
    {
        "_e2e",       // det: yolo26n_e2e
        "_seg_e2e",   // seg: yolo26n_seg_e2e
        "_pose_e2e",  // pose: yolo26n_pose_e2e
        "_cls",       // cls: yolo26n_cls (分类不需要e2e)
        "_obb_e2e"    // obb: yolo26n_obb_e2e
    };

    // 构建模型文件路径。
    // Detect task prefers a semantic SafeHat alias, then falls back to the legacy
    // yolo26n_e2e asset name to stay compatible with existing packaged assets.
    std::string modelstem;
    if (taskid == 0)
    {
        const char* detect_candidates[] = {"yolo26n_safehat", "yolo26n_e2e"};
        for (size_t i = 0; i < sizeof(detect_candidates) / sizeof(detect_candidates[0]); i++)
        {
            std::string candidate_param = std::string(detect_candidates[i]) + ".ncnn.param";
            std::string candidate_bin = std::string(detect_candidates[i]) + ".ncnn.bin";
            if (asset_exists(mgr, candidate_param.c_str()) && asset_exists(mgr, candidate_bin.c_str()))
            {
                modelstem = detect_candidates[i];
                break;
            }
        }
    }

    if (modelstem.empty())
        modelstem = std::string("yolo26n") + tasknames[(int)taskid];

    std::string parampath = modelstem + ".ncnn.param";
    std::string modelpath = modelstem + ".ncnn.bin";

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Loading YOLO26 model: %s", parampath.c_str());

    // Update global task id for latency logging in on_image_render
    g_current_taskid = (int)taskid;
    __android_log_print(ANDROID_LOG_INFO, "YOLO26BENCH",
        "LOAD task=%d name=%s model=%s",
        (int)taskid, taskid_to_name((int)taskid), parampath.c_str());
    
    bool use_gpu = (int)cpugpu == 1;
    bool use_turnip = (int)cpugpu == 2;

    // 转换细节排查阶段：强制CPU路径，排除移动端Vulkan/FP16对分数分布的干扰
    use_gpu = false;
    use_turnip = false;
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "ConversionDiag: force CPU inference path (cpugpu=%d)", (int)cpugpu);

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        {
            static int old_taskid = -1;
            static int old_cpugpu = -1;
            if (taskid != old_taskid || cpugpu != old_cpugpu)
            {
                // taskid or cpugpu changed
                delete g_yolo26;
                g_yolo26 = 0;
            }
            old_taskid = taskid;
            old_cpugpu = cpugpu;

            ncnn::destroy_gpu_instance();

            if (use_turnip)
            {
                ncnn::create_gpu_instance("libvulkan_freedreno.so");
            }
            else if (use_gpu)
            {
                ncnn::create_gpu_instance();
            }

            if (!g_yolo26)
            {
                if (taskid == 0) g_yolo26 = new YOLO26_det;
                if (taskid == 1) g_yolo26 = new YOLO26_seg;
                if (taskid == 2) g_yolo26 = new YOLO26_pose;
                if (taskid == 3) g_yolo26 = new YOLO26_cls;
                if (taskid == 4) g_yolo26 = new YOLO26_obb;

                g_yolo26->load(mgr, parampath.c_str(), modelpath.c_str(), use_gpu || use_turnip);

            }

            // Conversion diagnostic: run one offline synthetic frame through detector
            // to isolate camera input pipeline from model/runtime behavior.
            if (kConversionOfflineProbeEnabled && taskid == 0)
            {
                cv::Mat probe(480, 640, CV_8UC3);
                for (int y = 0; y < probe.rows; y++)
                {
                    unsigned char* row = probe.ptr<unsigned char>(y);
                    for (int x = 0; x < probe.cols; x++)
                    {
                        row[x * 3 + 0] = (unsigned char)((x * 37 + y * 17) & 0xff);
                        row[x * 3 + 1] = (unsigned char)((x * 13 + y * 29) & 0xff);
                        row[x * 3 + 2] = (unsigned char)(((x + y) * 11) & 0xff);
                    }
                }

                std::vector<Object> probe_objects;
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "OfflineProbe: begin synthetic frame detect");
                g_yolo26->detect(probe, probe_objects);
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "OfflineProbe: end synthetic frame detect objects=%zu", probe_objects.size());
            }
            
            // E2E模式固定使用640尺寸
            g_yolo26->set_det_target_size(640);
        }
    }

    return JNI_TRUE;
}

// public native boolean setDetectThresholds(float probThreshold, float nmsThreshold);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolo26ncnn_YOLO26Ncnn_setDetectThresholds(JNIEnv* env, jobject thiz, jfloat probThreshold, jfloat nmsThreshold)
{
    ncnn::MutexLockGuard g(lock);

    if (!g_yolo26)
        return JNI_FALSE;

    g_yolo26->set_det_thresholds((float)probThreshold, (float)nmsThreshold);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setDetectThresholds prob=%.2f nms=%.2f", (float)probThreshold, (float)nmsThreshold);

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolo26ncnn_YOLO26Ncnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_tencent_yolo26ncnn_YOLO26Ncnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolo26ncnn_YOLO26Ncnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

}
