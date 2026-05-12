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

// 1. install
//      pip3 install -U ultralytics pnnx ncnn
// 2. export yolo26-pose torchscript
//      yolo export model=yolo26n-pose.pt format=torchscript
// 3. convert torchscript with static shape
//      pnnx yolo26n-pose.torchscript
// 4. modify yolo26n_pose_pnnx.py for dynamic shape inference
//      A. modify reshape to support dynamic image sizes
//      B. permute tensor before concat and adjust concat axis
//      C. drop post-process part
//      before:
//          v_195 = v_194.view(1, 51, 6400)
//          v_201 = v_200.view(1, 51, 1600)
//          v_207 = v_206.view(1, 51, 400)
//          v_208 = torch.cat((v_195, v_201, v_207), dim=-1)
//          ...
//          v_254 = v_223.view(1, 65, 6400)
//          v_255 = v_238.view(1, 65, 1600)
//          v_256 = v_253.view(1, 65, 400)
//          v_257 = torch.cat((v_254, v_255, v_256), dim=2)
//          ...
//      after:
//          v_195 = v_194.view(1, 51, -1).transpose(1, 2)
//          v_201 = v_200.view(1, 51, -1).transpose(1, 2)
//          v_207 = v_206.view(1, 51, -1).transpose(1, 2)
//          v_208 = torch.cat((v_195, v_201, v_207), dim=1)
//          ...
//          v_254 = v_223.view(1, 65, -1).transpose(1, 2)
//          v_255 = v_238.view(1, 65, -1).transpose(1, 2)
//          v_256 = v_253.view(1, 65, -1).transpose(1, 2)
//          v_257 = torch.cat((v_254, v_255, v_256), dim=1)
//          return v_257, v_208
//      D. modify area attention for dynamic shape inference
//      before:
//          v_95 = self.model_10_m_0_attn_qkv_conv(v_94)
//          v_96 = v_95.view(1, 2, 128, 400)
//          v_97, v_98, v_99 = torch.split(tensor=v_96, dim=2, split_size_or_sections=(32,32,64))
//          v_100 = torch.transpose(input=v_97, dim0=-2, dim1=-1)
//          v_101 = torch.matmul(input=v_100, other=v_98)
//          v_102 = (v_101 * 0.176777)
//          v_103 = F.softmax(input=v_102, dim=-1)
//          v_104 = torch.transpose(input=v_103, dim0=-2, dim1=-1)
//          v_105 = torch.matmul(input=v_99, other=v_104)
//          v_106 = v_105.view(1, 128, 20, 20)
//          v_107 = v_99.reshape(1, 128, 20, 20)
//          v_108 = self.model_10_m_0_attn_pe_conv(v_107)
//          v_109 = (v_106 + v_108)
//          v_110 = self.model_10_m_0_attn_proj_conv(v_109)
//      after:
//          v_95 = self.model_10_m_0_attn_qkv_conv(v_94)
//          v_96 = v_95.view(1, 2, 128, -1)
//          v_97, v_98, v_99 = torch.split(tensor=v_96, dim=2, split_size_or_sections=(32,32,64))
//          v_100 = torch.transpose(input=v_97, dim0=-2, dim1=-1)
//          v_101 = torch.matmul(input=v_100, other=v_98)
//          v_102 = (v_101 * 0.176777)
//          v_103 = F.softmax(input=v_102, dim=-1)
//          v_104 = torch.transpose(input=v_103, dim0=-2, dim1=-1)
//          v_105 = torch.matmul(input=v_99, other=v_104)
//          v_106 = v_105.view(1, 128, v_95.size(2), v_95.size(3))
//          v_107 = v_99.reshape(1, 128, v_95.size(2), v_95.size(3))
//          v_108 = self.model_10_m_0_attn_pe_conv(v_107)
//          v_109 = (v_106 + v_108)
//          v_110 = self.model_10_m_0_attn_proj_conv(v_109)
// 5. re-export yolo26-pose torchscript
//      python3 -c 'import yolo26n_pose_pnnx; yolo26n_pose_pnnx.export_torchscript()'
// 6. convert new torchscript with dynamic shape
//      pnnx yolo26n_pose_pnnx.py.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320]
// 7. now you get ncnn model files
//      mv yolo26n_pose_pnnx.py.ncnn.param yolo26n_pose.ncnn.param
//      mv yolo26n_pose_pnnx.py.ncnn.bin yolo26n_pose.ncnn.bin

// the out blob would be a 2-dim tensor with w=65 h=8400
//
//        | bbox-reg 16 x 4       |score(1)|
//        +-----+-----+-----+-----+--------+
//        | dx0 | dy0 | dx1 | dy1 |   0.1  |
//   all /|     |     |     |     |        |
//  boxes |  .. |  .. |  .. |  .. |   0.0  |
//  (8400)|     |     |     |     |   .    |
//       \|     |     |     |     |   .    |
//        +-----+-----+-----+-----+--------+
//

//
//        | pose (51) |
//        +-----------+
//        |0.1........|
//   all /|           |
//  boxes |0.0........|
//  (8400)|     .     |
//       \|     .     |
//        +-----------+
//

#include "yolo26.h"

#include "layer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <float.h>
#include <stdio.h>
#include <vector>

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        // #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// YOLO26专用：无DFL的proposal生成（姿态估计任务）
static void generate_proposals_yolo26(const ncnn::Mat& pred, const ncnn::Mat& pred_points, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int num_points = pred_points.w / 3;  // 17 keypoints * 3 (x, y, conf)

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);
            const ncnn::Mat pred_points_grid = pred_points.row_range(y * num_grid_x + x, 1).reshape(3, num_points);

            // YOLO26 pose: 输出格式为 (8400, 5) = bbox(4) + score(1)
            int label = 0;
            // E2E模型已内置sigmoid，直接使用分数，避免双重sigmoid
            float score = pred_grid[4];

            if (score >= prob_threshold)
            {
                // YOLO26: 直接获取4个距离值
                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    pred_ltrb[k] = pred_grid[k] * stride;
                }

                float pb_cx = (x + 0.5f) * stride;
                float pb_cy = (y + 0.5f) * stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                std::vector<KeyPoint> keypoints;
                for (int k = 0; k < num_points; k++)
                {
                    KeyPoint keypoint;
                    keypoint.p.x = (x + pred_points_grid.row(k)[0] * 2) * stride;
                    keypoint.p.y = (y + pred_points_grid.row(k)[1] * 2) * stride;
                    // E2E模型已内置sigmoid，直接使用关键点可见性分数
                    keypoint.prob = pred_points_grid.row(k)[2];
                    keypoints.push_back(keypoint);
                }

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;
                obj.keypoints = keypoints;

                objects.push_back(obj);
            }
        }
    }
}

// YOLO26 (Legacy): 有DFL的proposal生成（兼容旧版本模型）
static void generate_proposals(const ncnn::Mat& pred, const ncnn::Mat& pred_points, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int reg_max_1 = 16;
    const int num_points = pred_points.w / 3;

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);
            const ncnn::Mat pred_points_grid = pred_points.row_range(y * num_grid_x + x, 1).reshape(3, num_points);

            // find label with max score
            int label = 0;
            float score = sigmoid(pred_grid[reg_max_1 * 4]);

            if (score >= prob_threshold)
            {
                ncnn::Mat pred_bbox = pred_grid.range(0, reg_max_1 * 4).reshape(reg_max_1, 4).clone();

                {
                    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);

                    softmax->forward_inplace(pred_bbox, opt);

                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = pred_bbox.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                    }

                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = (x + 0.5f) * stride;
                float pb_cy = (y + 0.5f) * stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                std::vector<KeyPoint> keypoints;
                for (int k = 0; k < num_points; k++)
                {
                    KeyPoint keypoint;
                    keypoint.p.x = (x + pred_points_grid.row(k)[0] * 2) * stride;
                    keypoint.p.y = (y + pred_points_grid.row(k)[1] * 2) * stride;
                    keypoint.prob = sigmoid(pred_points_grid.row(k)[2]);
                    keypoints.push_back(keypoint);
                }

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;
                obj.keypoints = keypoints;

                objects.push_back(obj);
            }
        }
    }
}

static void generate_proposals(const ncnn::Mat& pred, const ncnn::Mat& pred_points, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    int pred_row_offset = 0;
    for (size_t i = 0; i < strides.size(); i++)
    {
        const int stride = strides[i];

        const int num_grid_x = w / stride;
        const int num_grid_y = h / stride;
        const int num_grid = num_grid_x * num_grid_y;

        generate_proposals(pred.row_range(pred_row_offset, num_grid), pred_points.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects);

        pred_row_offset += num_grid;
    }
}

// YOLO26版本：无DFL
static void generate_proposals_yolo26(const ncnn::Mat& pred, const ncnn::Mat& pred_points, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    int pred_row_offset = 0;
    for (size_t i = 0; i < strides.size(); i++)
    {
        const int stride = strides[i];

        const int num_grid_x = w / stride;
        const int num_grid_y = h / stride;
        const int num_grid = num_grid_x * num_grid_y;

        generate_proposals_yolo26(pred.row_range(pred_row_offset, num_grid), pred_points.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects);

        pred_row_offset += num_grid;
    }
}

/**
 * 处理已解码的Pose输出（E2E模型，bbox和关键点已在像素坐标系）
 * 
 * 输出格式: (8400, 56) = [cx, cy, w, h, score, kp0_x, kp0_y, kp0_vis, ..., kp16_x, kp16_y, kp16_vis]
 * 所有坐标值已经是绝对像素坐标，分数已经过sigmoid
 * 仍需NMS（8400个proposals未做TopK筛选）
 *
 * @param out 模型输出 (8400, 56)
 * @param num_points 关键点数量 (17)
 * @param prob_threshold 置信度阈值
 * @param objects 输出检测结果
 */
static void process_decoded_pose_proposals(const ncnn::Mat& out, int num_points, float prob_threshold, std::vector<Object>& objects)
{
    for (int i = 0; i < out.h; i++)
    {
        const float* row = out.row(i);

        // 已解码格式: [cx, cy, w, h, score, kp...]
        float cx = row[0];
        float cy = row[1];
        float bw = row[2];
        float bh = row[3];
        float score = row[4];  // 已经过sigmoid

        if (score < prob_threshold)
            continue;

        Object obj;
        obj.rect.x = cx - bw * 0.5f;
        obj.rect.y = cy - bh * 0.5f;
        obj.rect.width = bw;
        obj.rect.height = bh;
        obj.label = 0;  // pose只有person类
        obj.prob = score;

        // 关键点已在像素坐标系，直接使用
        obj.keypoints.resize(num_points);
        for (int k = 0; k < num_points; k++)
        {
            int off = 5 + k * 3;
            obj.keypoints[k].p.x = row[off];
            obj.keypoints[k].p.y = row[off + 1];
            obj.keypoints[k].prob = row[off + 2];  // 已经过sigmoid
        }

        objects.push_back(obj);
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Decoded pose: %zu proposals above threshold %.2f", objects.size(), prob_threshold);
}

/**
 * YOLO26 End-to-End Pose 处理函数
 * 
 * E2E模式姿态输出格式:
 * - out0: (N, 300, 57) = [x, y, w, h, class_id, confidence, keypoints(17*3)]
 *   其中 keypoints 为 17个关键点，每个关键点3个值 [x, y, conf]
 * 
 * @param pred 检测输出 (300, 57)
 * @param prob_threshold 置信度阈值
 * @param objects 输出检测结果（包含关键点）
 */
static void process_e2e_pose_output(const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_detections = pred.h; // 300
    const int det_size = pred.w;       // 57 (6 + 17*3)
    const int num_keypoints = 17;
    
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Pose E2E mode: num_detections=%d, det_size=%d", num_detections, det_size);
    
    for (int i = 0; i < num_detections; i++)
    {
        const float* det = pred.row(i);
        
        // E2E姿态输出格式: [x_center, y_center, width, height, class_id, confidence, keypoints...]
        float x_center = det[0];
        float y_center = det[1];
        float w = det[2];
        float h = det[3];
        float class_id = det[4];
        float confidence = det[5];
        
        // 过滤低置信度检测
        if (confidence < prob_threshold)
            continue;
        
        Object obj;
        obj.rect.x = x_center - w * 0.5f;
        obj.rect.y = y_center - h * 0.5f;
        obj.rect.width = w;
        obj.rect.height = h;
        obj.label = (int)class_id;
        obj.prob = confidence;
        
        // 提取关键点
        obj.keypoints.resize(num_keypoints);
        for (int k = 0; k < num_keypoints; k++)
        {
            int offset = 6 + k * 3;
            obj.keypoints[k].p.x = det[offset];
            obj.keypoints[k].p.y = det[offset + 1];
            obj.keypoints[k].prob = det[offset + 2];
        }
        
        objects.push_back(obj);
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Pose E2E mode: detected %zu objects above threshold %.2f", objects.size(), prob_threshold);
}

int YOLO26_pose::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    const int target_size = det_target_size;//640;
    const float prob_threshold = det_prob_threshold;
    const float nms_threshold = det_nms_threshold;
    const float mask_threshold = 0.5f;

    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // ultralytics/cfg/models/yolo26.yaml
    std::vector<int> strides(3);
    strides[0] = 8;
    strides[1] = 16;
    strides[2] = 32;
    const int max_stride = 32;

    // letterbox pad to multiple of max_stride
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

    // letterbox pad to target_size square
    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolo26.create_extractor();

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    // 打印输出维度用于调试
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Pose Output shape: w=%d h=%d c=%d", out.w, out.h, out.c);

    // 处理转置格式: ncnn模型可能输出 (feature_dim × num_boxes) 即 w=num_boxes, h=feature_dim
    // 需要转为 (num_boxes × feature_dim) 即 w=feature_dim, h=num_boxes
    if (out.c == 1 && out.w > out.h && out.h > 1) {
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Pose: transposing out0 from w=%d h=%d to w=%d h=%d", out.w, out.h, out.h, out.w);
        ncnn::Mat out_t(out.h, out.w);
        for (int i = 0; i < out.w; i++) {
            float* dst = out_t.row(i);
            for (int j = 0; j < out.h; j++) {
                dst[j] = out.row(j)[i];
            }
        }
        out = out_t;
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Pose: after transpose: w=%d h=%d", out.w, out.h);
    }

    std::vector<Object> proposals;
    
    // E2E模式标志
    bool is_e2e_mode = false;
    int num_points = 17; // 默认17个关键点
    
    // 根据输出维度自动选择处理方式
    // YOLO26 E2E pose: w=57 (6+17*3), h<=300, 端到端无NMS
    // YOLO26 embedded: w=56 (5+17*3) 等, keypoints嵌入out0, 无需out1
    // YOLO26 separate: w=5 (4+1), keypoints在out1
    // YOLO26 (Legacy): w=65 (64+1), 使用DFL, keypoints在out1
    if (out.w == 57 && out.h <= 300) {
        // YOLO26 E2E模式姿态 (One-to-One Head)
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 E2E pose processing (no NMS needed)");
        is_e2e_mode = true;
        process_e2e_pose_output(out, prob_threshold, proposals);
    } else if (out.w > 5 && out.w != 65 && (out.w - 5) % 3 == 0) {
        // YOLO26 keypoints嵌入out0模式 (无DFL, 无需out1)
        // w = 4(bbox) + 1(score) + N*3(keypoints)  例如 56 = 4+1+17*3
        int kp_width = out.w - 5;
        num_points = kp_width / 3;

        // 检测是否为已解码格式（E2E模型的bbox已在像素坐标系）
        // 已解码: bbox列值在像素范围 (0~target_size)
        // 未解码: bbox列值在网格相对范围 (0~30)
        bool is_decoded_format = false;
        {
            float max_bbox_val = 0;
            for (int i = 0; i < std::min(out.h, 200); i++) {
                float v = fabsf(out.row(i)[0]);
                if (v > max_bbox_val) max_bbox_val = v;
            }
            is_decoded_format = (max_bbox_val > (float)target_size * 0.1f);
        }

        if (is_decoded_format) {
            // E2E已解码格式: bbox和关键点已在像素坐标系，直接使用
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using decoded pose processing (E2E, %d keypoints, max_bbox > %.0f)", num_points, (float)target_size * 0.1f);
            process_decoded_pose_proposals(out, num_points, prob_threshold, proposals);
        } else {
            // 原始格式: 需要网格相对解码
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 pose processing (embedded %d keypoints, no DFL)", num_points);
            ncnn::Mat pred(5, out.h);
            ncnn::Mat pred_points(kp_width, out.h);
            for (int i = 0; i < out.h; i++) {
                const float* src = out.row(i);
                memcpy(pred.row(i), src, 5 * sizeof(float));
                memcpy(pred_points.row(i), src + 5, kp_width * sizeof(float));
            }
            generate_proposals_yolo26(pred, pred_points, strides, in_pad, prob_threshold, proposals);
        }
    } else if (out.w == 5) {
        // YOLO26 One-to-Many模式 (无DFL, keypoints在out1)
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 pose processing (separate keypoints, no DFL)");
        ncnn::Mat out_points;
        ex.extract("out1", out_points);
        if (out_points.empty()) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Pose: failed to extract out1 for keypoints");
            return -1;
        }
        generate_proposals_yolo26(out, out_points, strides, in_pad, prob_threshold, proposals);
        num_points = out_points.w / 3;
    } else {
        // YOLO26 (Legacy) 模式 (有DFL, keypoints在out1)
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 (Legacy) pose processing (with DFL)");
        ncnn::Mat out_points;
        ex.extract("out1", out_points);
        if (out_points.empty()) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Pose: failed to extract out1 for YOLO26 Legacy mode");
            return -1;
        }
        generate_proposals(out, out_points, strides, in_pad, prob_threshold, proposals);
        num_points = out_points.w / 3;
    }

    int count;
    std::vector<int> picked;
    
    if (is_e2e_mode) {
        // E2E模式无需NMS
        count = proposals.size();
        picked.resize(count);
        for (int i = 0; i < count; i++) {
            picked[i] = i;
        }
    } else {
        // 传统模式需要排序和NMS
        qsort_descent_inplace(proposals);
        nms_sorted_bboxes(proposals, picked, nms_threshold);
        count = picked.size();
    }
    if (count == 0)
        return 0;

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        for (int j = 0; j < num_points; j++)
        {
            objects[i].keypoints[j].p.x = (objects[i].keypoints[j].p.x - (wpad / 2)) / scale;
            objects[i].keypoints[j].p.y = (objects[i].keypoints[j].p.y - (hpad / 2)) / scale;
        }

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // sort objects by area
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);

    return 0;
}

int YOLO26_pose::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"person"};

    static const cv::Scalar colors[] = {
        cv::Scalar( 67,  54, 244),
        cv::Scalar( 30,  99, 233),
        cv::Scalar( 39, 176, 156),
        cv::Scalar( 58, 183, 103),
        cv::Scalar( 81, 181,  63),
        cv::Scalar(150, 243,  33),
        cv::Scalar(169, 244,   3),
        cv::Scalar(188, 212,   0),
        cv::Scalar(150, 136,   0),
        cv::Scalar(175,  80,  76),
        cv::Scalar(195,  74, 139),
        cv::Scalar(220,  57, 205),
        cv::Scalar(235,  59, 255),
        cv::Scalar(193,   7, 255),
        cv::Scalar(152,   0, 255),
        cv::Scalar( 87,  34, 255),
        cv::Scalar( 85,  72, 121),
        cv::Scalar(158, 158, 158),
        cv::Scalar(125, 139,  96)
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[i % 19];

        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                // obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        // draw bone
        static const int joint_pairs[16][2] = {
            {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}
        };
        static const cv::Scalar bone_colors[] = {
            cv::Scalar(  0,   0, 255),
            cv::Scalar(  0,   0, 255),
            cv::Scalar(  0,   0, 255),
            cv::Scalar(  0,   0, 255),
            cv::Scalar(  0, 255, 128),
            cv::Scalar(  0, 255, 128),
            cv::Scalar(  0, 255, 128),
            cv::Scalar(  0, 255, 128),
            cv::Scalar(  0, 255, 128),
            cv::Scalar(255, 255,  51),
            cv::Scalar(255, 255,  51),
            cv::Scalar(255, 255,  51),
            cv::Scalar(255,  51, 153),
            cv::Scalar(255,  51, 153),
            cv::Scalar(255,  51, 153),
            cv::Scalar(255,  51, 153),
        };

        for (int j = 0; j < 16; j++)
        {
            const KeyPoint& p1 = obj.keypoints[joint_pairs[j][0]];
            const KeyPoint& p2 = obj.keypoints[joint_pairs[j][1]];

            if (p1.prob < 0.2f || p2.prob < 0.2f)
                continue;

            cv::line(rgb, p1.p, p2.p, bone_colors[j], 2);
        }

        // draw joint
        for (size_t j = 0; j < obj.keypoints.size(); j++)
        {
            const KeyPoint& keypoint = obj.keypoints[j];

            // fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);

            if (keypoint.prob < 0.2f)
                continue;

            cv::circle(rgb, keypoint.p, 3, color, -1);
        }

        cv::rectangle(rgb, obj.rect, color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return 0;
}
