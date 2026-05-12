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

#include "yolo26.h"

YOLO26::YOLO26()
{
    det_target_size = 640;
    det_prob_threshold = 0.35f;  // 激进校准后 calibrate(0.999)≈0.66, 正常检测分数在0.3-0.65
    det_nms_threshold = 0.30f;
}

YOLO26::~YOLO26()
{
}

int YOLO26::load(const char* parampath, const char* modelpath, bool use_gpu)
{
    yolo26.clear();

    yolo26.opt = ncnn::Option();

    // Keep tensor precision consistent with post-processing code that reads float outputs.
    // On some ARM devices, fp16 storage/arithmetic may be enabled by default and can
    // cause raw pointer float reads (out.row()) to interpret fp16 buffers incorrectly.
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

    // Keep tensor precision consistent with post-processing code that reads float outputs.
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
