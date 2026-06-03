#include "yolo26.h"

YOLO26::YOLO26()
{
    det_target_size = 640;
    det_prob_threshold = 0.35f;
    det_nms_threshold = 0.30f;
}

YOLO26::~YOLO26()
{
}

int YOLO26::load(const char* parampath, const char* modelpath, bool use_gpu)
{
    yolo26.clear();

    yolo26.opt = ncnn::Option();

    yolo26.opt.use_fp16_packed = false;
    yolo26.opt.use_fp16_storage = false;
    yolo26.opt.use_fp16_arithmetic = false;
    yolo26.opt.use_bf16_storage = false;
    yolo26.opt.use_packing_layout = false;

#if NCNN_VULKAN
    yolo26.opt.use_vulkan_compute = use_gpu;
#endif

    yolo26.load_param(parampath);
    yolo26.load_model(modelpath);

    return 0;
}

int YOLO26::load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu)
{
    yolo26.clear();

    yolo26.opt = ncnn::Option();

    yolo26.opt.use_fp16_packed = false;
    yolo26.opt.use_fp16_storage = false;
    yolo26.opt.use_fp16_arithmetic = false;
    yolo26.opt.use_bf16_storage = false;
    yolo26.opt.use_packing_layout = false;

#if NCNN_VULKAN
    yolo26.opt.use_vulkan_compute = use_gpu;
#endif

    yolo26.load_param(mgr, parampath);
    yolo26.load_model(mgr, modelpath);

    return 0;
}

void YOLO26::set_det_target_size(int target_size)
{
    det_target_size = target_size;
}

void YOLO26::set_det_thresholds(float prob_threshold, float nms_threshold)
{
    if (prob_threshold < 0.01f) prob_threshold = 0.01f;
    if (prob_threshold > 0.99f) prob_threshold = 0.99f;

    if (nms_threshold < 0.05f) nms_threshold = 0.05f;
    if (nms_threshold > 0.95f) nms_threshold = 0.95f;

    det_prob_threshold = prob_threshold;
    det_nms_threshold = nms_threshold;
}
