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
// 2. export yolo26-obb torchscript
//      yolo export model=yolo26n-obb.pt format=torchscript
// 3. convert torchscript with static shape
//      pnnx yolo26n-obb.torchscript
// 4. modify yolo26n_obb_pnnx.py for dynamic shape inference
//      A. modify reshape to support dynamic image sizes
//      B. permute tensor before concat and adjust concat axis
//      C. drop post-process part
//      before:
//          v_195 = v_194.view(1, 1, 16384)
//          v_201 = v_200.view(1, 1, 4096)
//          v_207 = v_206.view(1, 1, 1024)
//          v_208 = torch.cat((v_195, v_201, v_207), dim=2)
//          ...
//          v_256 = v_225.view(1, 79, 16384)
//          v_257 = v_240.view(1, 79, 4096)
//          v_258 = v_255.view(1, 79, 1024)
//          v_259 = torch.cat((v_256, v_257, v_258), dim=2)
//          ...
//      after:
//          v_195 = v_194.view(1, 1, -1).transpose(1, 2)
//          v_201 = v_200.view(1, 1, -1).transpose(1, 2)
//          v_207 = v_206.view(1, 1, -1).transpose(1, 2)
//          v_208 = torch.cat((v_195, v_201, v_207), dim=1)
//          ...
//          v_256 = v_225.view(1, 79, -1).transpose(1, 2)
//          v_257 = v_240.view(1, 79, -1).transpose(1, 2)
//          v_258 = v_255.view(1, 79, -1).transpose(1, 2)
//          v_259 = torch.cat((v_256, v_257, v_258), dim=1)
//          return v_259, v_208
//      D. modify area attention for dynamic shape inference
//      before:
//          v_95 = self.model_10_m_0_attn_qkv_conv(v_94)
//          v_96 = v_95.view(1, 2, 128, 1024)
//          v_97, v_98, v_99 = torch.split(tensor=v_96, dim=2, split_size_or_sections=(32,32,64))
//          v_100 = torch.transpose(input=v_97, dim0=-2, dim1=-1)
//          v_101 = torch.matmul(input=v_100, other=v_98)
//          v_102 = (v_101 * 0.176777)
//          v_103 = F.softmax(input=v_102, dim=-1)
//          v_104 = torch.transpose(input=v_103, dim0=-2, dim1=-1)
//          v_105 = torch.matmul(input=v_99, other=v_104)
//          v_106 = v_105.view(1, 128, 32, 32)
//          v_107 = v_99.reshape(1, 128, 32, 32)
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
// 5. re-export yolo26-obb torchscript
//      python3 -c 'import yolo26n_obb_pnnx; yolo26n_obb_pnnx.export_torchscript()'
// 6. convert new torchscript with dynamic shape
//      pnnx yolo26n_obb_pnnx.py.pt inputshape=[1,3,1024,1024] inputshape2=[1,3,512,512]
// 7. now you get ncnn model files
//      mv yolo26n_obb_pnnx.py.ncnn.param yolo26n_obb.ncnn.param
//      mv yolo26n_obb_pnnx.py.ncnn.bin yolo26n_obb.ncnn.bin

// the out blob would be a 2-dim tensor with w=79 h=21504
//
//        | bbox-reg 16 x 4       |score(15)|
//        +-----+-----+-----+-----+---------+
//        | dx0 | dy0 | dx1 | dy1 | 0.1 ... |
//   all /|     |     |     |     |     ... |
//  boxes |  .. |  .. |  .. |  .. | 0.0 ... |
// (21504)|     |     |     |     |  .  ... |
//       \|     |     |     |     |  .  ... |
//        +-----+-----+-----+-----+---------+
//

// the out blob would be a 2-dim tensor with w=1 h=21504
//
//        | degree(1)|
//        +----------+
//        |    0.1   |
//   all /|          |
//  boxes |    0.0   |
// (21504)|     .    |
//       \|     .    |
//        +----------+
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
    std::vector<cv::Point2f> intersection;
    cv::rotatedRectangleIntersection(a.rrect, b.rrect, intersection);
    if (intersection.empty())
        return 0.f;

    return cv::contourArea(intersection);
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
        areas[i] = objects[i].rrect.size.area();
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
            // float IoU = inter_area / union_area;
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

// YOLO26专用：无DFL的proposal生成（旋转目标检测任务）
static void generate_proposals_yolo26(const ncnn::Mat& pred, const ncnn::Mat& pred_angle, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    // YOLO26 obb: 输出格式为 (8400, 19) = bbox(4) + classes(15)
    const int num_class = pred.w - 4; // 15 for DOTAv1

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);

            int label = -1;
            float score = -FLT_MAX;
            {
                const ncnn::Mat pred_score = pred_grid.range(4, num_class);

                for (int k = 0; k < num_class; k++)
                {
                    float s = pred_score[k];
                    if (s > score)
                    {
                        label = k;
                        score = s;
                    }
                }

                // score = sigmoid(score);  // 已移除 - E2E模型已内置sigmoid，避免双重sigmoid
            }

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

                const float angle = sigmoid(pred_angle.row(y * num_grid_x + x)[0]) - 0.25f;

                const float angle_rad = angle * 3.14159265358979323846f;
                const float angle_degree = angle * 180.f;

                float cos = cosf(angle_rad);
                float sin = sinf(angle_rad);

                float lt_x = pb_cx - pred_ltrb[0] * cos + pred_ltrb[1] * sin;
                float lt_y = pb_cy - pred_ltrb[0] * sin - pred_ltrb[1] * cos;
                float rt_x = pb_cx + pred_ltrb[2] * cos + pred_ltrb[1] * sin;
                float rt_y = pb_cy + pred_ltrb[2] * sin - pred_ltrb[1] * cos;
                float rb_x = pb_cx + pred_ltrb[2] * cos - pred_ltrb[3] * sin;
                float rb_y = pb_cy + pred_ltrb[2] * sin + pred_ltrb[3] * cos;
                float lb_x = pb_cx - pred_ltrb[0] * cos - pred_ltrb[3] * sin;
                float lb_y = pb_cy - pred_ltrb[0] * sin + pred_ltrb[3] * cos;

                float c_x = (lt_x + rt_x + rb_x + lb_x) / 4.f;
                float c_y = (lt_y + rt_y + rb_y + lb_y) / 4.f;

                float ow = sqrtf((rt_x - lt_x) * (rt_x - lt_x) + (rt_y - lt_y) * (rt_y - lt_y));
                float oh = sqrtf((rb_x - rt_x) * (rb_x - rt_x) + (rb_y - rt_y) * (rb_y - rt_y));

                Object obj;
                obj.rrect.center.x = c_x;
                obj.rrect.center.y = c_y;
                obj.rrect.size.width = ow;
                obj.rrect.size.height = oh;
                obj.rrect.angle = angle_degree;
                obj.label = label;
                obj.prob = score;

                objects.push_back(obj);
            }
        }
    }
}

// YOLO26 (Legacy): 有DFL的proposal生成（兼容旧版本模型）
static void generate_proposals(const ncnn::Mat& pred, const ncnn::Mat& pred_angle, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int reg_max_1 = 16;
    const int num_class = pred.w - reg_max_1 * 4; // number of classes. 15 for DOTAv1

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);

            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            {
                const ncnn::Mat pred_score = pred_grid.range(reg_max_1 * 4, num_class);

                for (int k = 0; k < num_class; k++)
                {
                    float s = pred_score[k];
                    if (s > score)
                    {
                        label = k;
                        score = s;
                    }
                }

                score = sigmoid(score);
            }

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

                const float angle = sigmoid(pred_angle.row(y * num_grid_x + x)[0]) - 0.25f;

                const float angle_rad = angle * 3.14159265358979323846f;
                const float angle_degree = angle * 180.f;

                float cos = cosf(angle_rad);
                float sin = sinf(angle_rad);

                float xx = (pred_ltrb[2] - pred_ltrb[0]) * 0.5f;
                float yy = (pred_ltrb[3] - pred_ltrb[1]) * 0.5f;
                float xr = xx * cos - yy * sin;
                float yr = xx * sin + yy * cos;
                const float cx = pb_cx + xr;
                const float cy = pb_cy + yr;
                const float ww = pred_ltrb[2] + pred_ltrb[0];
                const float hh = pred_ltrb[3] + pred_ltrb[1];

                Object obj;
                obj.rrect = cv::RotatedRect(cv::Point2f(cx, cy), cv::Size_<float>(ww, hh), angle_degree);
                obj.label = label;
                obj.prob = score;

                objects.push_back(obj);
            }
        }
    }
}

static void generate_proposals(const ncnn::Mat& pred, const ncnn::Mat& pred_angle, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
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

        generate_proposals(pred.row_range(pred_row_offset, num_grid), pred_angle.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects);

        pred_row_offset += num_grid;
    }
}

// YOLO26版本：无DFL
static void generate_proposals_yolo26(const ncnn::Mat& pred, const ncnn::Mat& pred_angle, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
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

        generate_proposals_yolo26(pred.row_range(pred_row_offset, num_grid), pred_angle.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects);

        pred_row_offset += num_grid;
    }
}

/**
 * 处理已解码的OBB输出（E2E模型，bbox已在像素坐标系）
 * 
 * 输出格式: (8400, 4+nc+1) = [cx, cy, w, h, class_0, ..., class_nc-1, angle_raw]
 * 由ncnn param文件追溯:
 *   out0 = cat(cat(decoded_bbox(4), sigmoid(class_scores)(nc)), raw_angle(1))
 * - bbox: 列0-3, 已解码为绝对像素坐标
 * - class scores: 列4 ~ 列(w-2), 已经过sigmoid
 * - angle: 最后一列(w-1), 原始值需要sigmoid
 * 仍需NMS（8400个proposals未做TopK筛选）
 *
 * @param out 模型输出 (8400, 20) for DOTAv1 (15 classes)
 * @param prob_threshold 置信度阈值
 * @param objects 输出检测结果
 */
static void process_decoded_obb_proposals(const ncnn::Mat& out, float prob_threshold, std::vector<Object>& objects)
{
    const int num_class = out.w - 4 - 1;  // 减去 bbox(4) 和 angle(1)

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Decoded OBB: h=%d w=%d num_class=%d", out.h, out.w, num_class);

    for (int i = 0; i < out.h; i++)
    {
        const float* row = out.row(i);

        float cx = row[0];
        float cy = row[1];
        float bw = row[2];
        float bh = row[3];

        // class scores 在列4 ~ 列(w-2)，已经过sigmoid，直接比较
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float s = row[4 + k];  // 已sigmoid，无需再次sigmoid
            if (s > score)
            {
                label = k;
                score = s;
            }
        }

        if (score < prob_threshold)
            continue;

        // angle 在最后一列，是raw值，需要sigmoid
        float angle_raw = row[out.w - 1];
        float angle_sigmoid_val = sigmoid(angle_raw);
        // sigmoid(0~1) → normalized(-0.25~0.75) → degrees(-45°~135°)
        float angle = angle_sigmoid_val - 0.25f;
        float angle_degree = angle * 180.f;

        Object obj;
        obj.rrect = cv::RotatedRect(cv::Point2f(cx, cy), cv::Size_<float>(bw, bh), angle_degree);
        obj.label = label;
        obj.prob = score;

        objects.push_back(obj);
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Decoded OBB: %zu proposals above threshold %.2f", objects.size(), prob_threshold);
}

/**
 * YOLO26 End-to-End OBB 处理函数
 * 
 * E2E模式旋转框输出格式:
 * - out0: (N, 300, 7) = [x_center, y_center, width, height, angle, class_id, confidence]
 * 
 * 注意：E2E模式下angle已经是弧度值，需要转换为角度
 * 
 * @param pred 检测输出 (300, 7)
 * @param prob_threshold 置信度阈值
 * @param objects 输出检测结果
 */
static void process_e2e_obb_output(const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_detections = pred.h; // 300
    const int det_size = pred.w;       // 7
    
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "OBB E2E mode: num_detections=%d, det_size=%d", num_detections, det_size);
    
    for (int i = 0; i < num_detections; i++)
    {
        const float* det = pred.row(i);
        
        // E2E OBB输出格式: [x_center, y_center, width, height, angle, class_id, confidence]
        float x_center = det[0];
        float y_center = det[1];
        float w = det[2];
        float h = det[3];
        float angle = det[4];          // 弧度值
        float class_id = det[5];
        float confidence = det[6];
        
        // 过滤低置信度检测
        if (confidence < prob_threshold)
            continue;
        
        // 转换弧度为角度
        float angle_degree = angle * 180.f / 3.14159265358979323846f;
        
        Object obj;
        obj.rrect = cv::RotatedRect(cv::Point2f(x_center, y_center), cv::Size_<float>(w, h), angle_degree);
        obj.label = (int)class_id;
        obj.prob = confidence;
        
        objects.push_back(obj);
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "OBB E2E mode: detected %zu objects above threshold %.2f", objects.size(), prob_threshold);
}
int YOLO26_obb::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    const int target_size = det_target_size;//1024;
    const float prob_threshold = det_prob_threshold;
    const float nms_threshold = det_nms_threshold;

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
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "OBB Output shape: w=%d h=%d c=%d", out.w, out.h, out.c);

    // 处理转置格式: ncnn模型可能输出 (feature_dim × num_boxes) 即 w=num_boxes, h=feature_dim
    // 需要转为 (num_boxes × feature_dim) 即 w=feature_dim, h=num_boxes
    if (out.c == 1 && out.w > out.h && out.h > 1) {
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "OBB: transposing out0 from w=%d h=%d to w=%d h=%d", out.w, out.h, out.h, out.w);
        ncnn::Mat out_t(out.h, out.w);
        for (int i = 0; i < out.w; i++) {
            float* dst = out_t.row(i);
            for (int j = 0; j < out.h; j++) {
                dst[j] = out.row(j)[i];
            }
        }
        out = out_t;
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "OBB: after transpose: w=%d h=%d", out.w, out.h);
    }

    std::vector<Object> proposals;
    
    // E2E模式标志
    bool is_e2e_mode = false;
    
    // 检查模型是否有out1 blob (用于区分嵌入式和分离式angle格式)
    // 打印模型输入/输出名称用于调试
    {
        const std::vector<ncnn::Blob>& blobs = yolo26.blobs();
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "model output names:");
        for (size_t i = 0; i < blobs.size(); i++) {
            if (blobs[i].name.substr(0, 3) == "out") {
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "  %s", blobs[i].name.c_str());
            }
        }
    }

    bool has_out1 = false;
    {
        const std::vector<ncnn::Blob>& blobs = yolo26.blobs();
        for (size_t i = 0; i < blobs.size(); i++) {
            if (blobs[i].name == "out1") {
                has_out1 = true;
                break;
            }
        }
    }

    // 根据输出维度自动选择处理方式
    // YOLO26 E2E obb: w=7, h<=300, 端到端无NMS
    // YOLO26 embedded: angle嵌入out0最后一列 (w = 4+num_class+1, 如 20 for DOTAv1)
    // YOLO26 separate: angle在out1 (w = 4+num_class, 如 19 for DOTAv1)
    // YOLO26 (Legacy): w>=65, 使用DFL, angle在out1
    if (out.w == 7 && out.h <= 300) {
        // YOLO26 E2E模式旋转框 (One-to-One Head)
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 E2E obb processing (no NMS needed)");
        is_e2e_mode = true;
        process_e2e_obb_output(out, prob_threshold, proposals);
    } else if (has_out1) {
        // 模型有out1 → angle在单独的out1中
        ncnn::Mat out_angle;
        ex.extract("out1", out_angle);
        if (out.w <= 32) {
            // YOLO26 (无DFL)
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 obb processing (separate angle, no DFL)");
            generate_proposals_yolo26(out, out_angle, strides, in_pad, prob_threshold, proposals);
        } else {
            // YOLO26 (Legacy) (有DFL)
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 (Legacy) obb processing (with DFL)");
            generate_proposals(out, out_angle, strides, in_pad, prob_threshold, proposals);
        }
    } else {
        // 模型没有out1 → angle嵌入在out0中
        // 检测是否为已解码格式（E2E模型的bbox已在像素坐标系）
        // E2E已解码: 列布局 = [cx, cy, w, h, angle_sigmoid, class_0, ..., class_nc-1]
        // 原始未解码: 列布局 = [ltrb(4), classes(nc), angle(1)]  (angle在最后一列)
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
            // E2E已解码格式: [cx, cy, w, h, angle, classes...] 全部已在像素坐标系
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using decoded OBB processing (E2E, w=%d)", out.w);
            process_decoded_obb_proposals(out, prob_threshold, proposals);
        } else {
            // 原始格式: angle嵌入在out0最后一列
            // 分离: pred = 前w-1列 (bbox+classes), angle = 最后1列
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 obb processing (embedded angle, no DFL, w=%d)", out.w);
            int pred_width = out.w - 1;
            ncnn::Mat pred(pred_width, out.h);
            ncnn::Mat pred_angle(1, out.h);
            for (int i = 0; i < out.h; i++) {
                const float* src = out.row(i);
                memcpy(pred.row(i), src, pred_width * sizeof(float));
                pred_angle.row(i)[0] = src[pred_width];
            }
            generate_proposals_yolo26(pred, pred_angle, strides, in_pad, prob_threshold, proposals);
        }
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
        Object obj = proposals[picked[i]];

        // adjust offset to original unpadded
        obj.rrect.center.x = (obj.rrect.center.x - (wpad / 2)) / scale;
        obj.rrect.center.y = (obj.rrect.center.y - (hpad / 2)) / scale;
        obj.rrect.size.width = (obj.rrect.size.width) / scale;
        obj.rrect.size.height = (obj.rrect.size.height) / scale;

        objects[i] = obj;
    }

    return 0;
}

int YOLO26_obb::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "plane", "ship", "storage tank", "baseball diamond", "tennis court",
        "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
        "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"
    };

    static const cv::Scalar colors[] = {
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

        const cv::Scalar& color = colors[obj.label];

        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f  @ %.2f\n", obj.label, obj.prob,
                // obj.rrect.center.x, obj.rrect.center.y, obj.rrect.size.width, obj.rrect.size.height, obj.rrect.angle);

        cv::Point2f corners[4];
        obj.rrect.points(corners);
        cv::line(rgb, corners[0], corners[1], color);
        cv::line(rgb, corners[1], corners[2], color);
        cv::line(rgb, corners[2], corners[3], color);
        cv::line(rgb, corners[3], corners[0], color);
    }

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[obj.label];

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rrect.center.x - label_size.width / 2;
        int y = obj.rrect.center.y - label_size.height / 2 - baseLine;
        if (y < 0)
            y = 0;
        if (y + label_size.height > rgb.rows)
            y = rgb.rows - label_size.height;
        if (x < 0)
            x = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return 0;
}
