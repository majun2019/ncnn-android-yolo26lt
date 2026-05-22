#include "yolo26.h"

#include "layer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if defined(__ANDROID__) || defined(ANDROID)
#include <android/log.h>
#else

#define ANDROID_LOG_DEBUG 3
#define ANDROID_LOG_ERROR 6
static inline int __android_log_print(int, const char*, const char*, ...) { return 0; }
#endif

#include <algorithm>
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

            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    {

        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }

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

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;

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

static void generate_proposals_yolo26(const ncnn::Mat& pred, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int num_class = pred.w - 4 - 32;

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

            }

            if (score >= prob_threshold)
            {

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

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;
                obj.gindex = y * num_grid_x + x;

                objects.push_back(obj);
            }
        }
    }
}

static void generate_proposals(const ncnn::Mat& pred, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int reg_max_1 = 16;
    const int num_class = pred.w - reg_max_1 * 4;

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);

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
                    pd.set(0, 1);
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

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;
                obj.gindex = y * num_grid_x + x;

                objects.push_back(obj);
            }
        }
    }
}

static void generate_proposals(const ncnn::Mat& pred, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
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

        std::vector<Object> objects_stride;
        generate_proposals(pred.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects_stride);

        for (size_t j = 0; j < objects_stride.size(); j++)
        {
            Object obj = objects_stride[j];
            obj.gindex += pred_row_offset;
            objects.push_back(obj);
        }

        pred_row_offset += num_grid;
    }
}

static void generate_proposals_yolo26(const ncnn::Mat& pred, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
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

        std::vector<Object> objects_stride;
        generate_proposals_yolo26(pred.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects_stride);

        for (size_t j = 0; j < objects_stride.size(); j++)
        {
            Object obj = objects_stride[j];
            obj.gindex += pred_row_offset;
            objects.push_back(obj);
        }

        pred_row_offset += num_grid;
    }
}

static void process_e2e_seg_output(const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_detections = pred.h;
    const int det_size = pred.w;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Seg E2E mode: num_detections=%d, det_size=%d", num_detections, det_size);

    for (int i = 0; i < num_detections; i++)
    {
        const float* det = pred.row(i);

        float x_center = det[0];
        float y_center = det[1];
        float w = det[2];
        float h = det[3];
        float class_id = det[4];
        float confidence = det[5];

        if (confidence < prob_threshold)
            continue;

        Object obj;
        obj.rect.x = x_center - w * 0.5f;
        obj.rect.y = y_center - h * 0.5f;
        obj.rect.width = w;
        obj.rect.height = h;
        obj.label = (int)class_id;
        obj.prob = confidence;
        obj.gindex = i;

        objects.push_back(obj);
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Seg E2E mode: detected %zu objects above threshold %.2f", objects.size(), prob_threshold);
}

int YOLO26_seg::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    const int target_size = det_target_size;
    const float prob_threshold = det_prob_threshold;
    const float nms_threshold = det_nms_threshold;
    const float mask_threshold = 0.5f;

    int img_w = rgb.cols;
    int img_h = rgb.rows;

    std::vector<int> strides(3);
    strides[0] = 8;
    strides[1] = 16;
    strides[2] = 32;
    const int max_stride = 32;

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

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Seg Output shape: w=%d h=%d c=%d", out.w, out.h, out.c);

    if (out.c == 1 && out.w > out.h && out.h > 1) {
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Seg: transposing out0 from w=%d h=%d to w=%d h=%d", out.w, out.h, out.h, out.w);
        ncnn::Mat out_t(out.h, out.w);
        for (int i = 0; i < out.w; i++) {
            float* dst = out_t.row(i);
            for (int j = 0; j < out.h; j++) {
                dst[j] = out.row(j)[i];
            }
        }
        out = out_t;
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Seg: after transpose: w=%d h=%d", out.w, out.h);
    }

    std::vector<Object> proposals;

    bool is_e2e_mode = false;
    bool is_yolo26_mode = false;

    if (out.w == 38 && out.h <= 300) {

        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 E2E seg processing (no NMS needed)");
        is_e2e_mode = true;
        process_e2e_seg_output(out, prob_threshold, proposals);
    } else if (out.w == 116) {

        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 seg processing (no DFL)");
        is_yolo26_mode = true;
        generate_proposals_yolo26(out, strides, in_pad, prob_threshold, proposals);
    } else {

        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 (Legacy) seg processing (with DFL)");
        generate_proposals(out, strides, in_pad, prob_threshold, proposals);
    }

    int count;
    std::vector<int> picked;

    if (is_e2e_mode) {

        count = proposals.size();
        picked.resize(count);
        for (int i = 0; i < count; i++) {
            picked[i] = i;
        }
    } else {

        qsort_descent_inplace(proposals);

        nms_sorted_bboxes(proposals, picked, nms_threshold);
        count = picked.size();
    }
    if (count == 0)
        return 0;

    ncnn::Mat mask_feat;
    ncnn::Mat mask_protos;

    if (is_e2e_mode) {

        ex.extract("out1", mask_protos);

        if (mask_protos.empty()) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Seg E2E: failed to extract out1 (mask_protos)!");
            return -1;
        }
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Seg E2E mask_protos: w=%d h=%d c=%d", mask_protos.w, mask_protos.h, mask_protos.c);
    } else if (is_yolo26_mode) {

        ex.extract("out1", mask_protos);
        if (mask_protos.empty()) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Seg YOLO26: failed to extract out1 (mask_protos)!");
            return -1;
        }
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Seg YOLO26 mask_protos: w=%d h=%d c=%d", mask_protos.w, mask_protos.h, mask_protos.c);
    } else {

        ex.extract("out1", mask_feat);
        ex.extract("out2", mask_protos);

        if (mask_feat.empty() || mask_protos.empty()) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Seg legacy: failed to extract out1/out2!");
            return -1;
        }
    }

    const int mask_coeff_width = (is_e2e_mode || is_yolo26_mode) ? 32 : mask_feat.w;
    ncnn::Mat objects_mask_feat(mask_coeff_width, 1, count);

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        x0 = (std::max)((std::min)(x0, (float)(img_w - 1)), 0.f);
        y0 = (std::max)((std::min)(y0, (float)(img_h - 1)), 0.f);
        x1 = (std::max)((std::min)(x1, (float)(img_w - 1)), 0.f);
        y1 = (std::max)((std::min)(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

        if (is_e2e_mode) {

            const float* det = out.row(objects[i].gindex);
            memcpy(objects_mask_feat.channel(i), det + 6, mask_coeff_width * sizeof(float));
        } else if (is_yolo26_mode) {

            const float* box_row = out.row(objects[i].gindex);
            memcpy(objects_mask_feat.channel(i), box_row + (out.w - 32), mask_coeff_width * sizeof(float));
        } else {

            memcpy(objects_mask_feat.channel(i), mask_feat.row(objects[i].gindex), mask_coeff_width * sizeof(float));
        }
    }

    ncnn::Mat objects_mask;
    {
        ncnn::Layer* gemm = ncnn::create_layer("Gemm");

        ncnn::ParamDict pd;
        pd.set(6, 1);
        pd.set(7, count);
        pd.set(8, mask_protos.w * mask_protos.h);
        pd.set(9, mask_coeff_width);
        pd.set(10, -1);
        pd.set(11, 1);
        gemm->load_param(pd);

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;

        gemm->create_pipeline(opt);

        std::vector<ncnn::Mat> gemm_inputs(2);
        gemm_inputs[0] = objects_mask_feat;
        gemm_inputs[1] = mask_protos.reshape(mask_protos.w * mask_protos.h, 1, mask_protos.c);
        std::vector<ncnn::Mat> gemm_outputs(1);
        gemm->forward(gemm_inputs, gemm_outputs, opt);
        objects_mask = gemm_outputs[0].reshape(mask_protos.w, mask_protos.h, count);

        gemm->destroy_pipeline(opt);

        delete gemm;
    }
    {
        ncnn::Layer* sigmoid = ncnn::create_layer("Sigmoid");

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;

        sigmoid->create_pipeline(opt);

        sigmoid->forward_inplace(objects_mask, opt);

        sigmoid->destroy_pipeline(opt);

        delete sigmoid;
    }

    {
        ncnn::Mat objects_mask_resized;
        ncnn::resize_bilinear(objects_mask, objects_mask_resized, in_pad.w / scale, in_pad.h / scale);
        objects_mask = objects_mask_resized;
    }

    for (int i = 0; i < count; i++)
    {
        Object& obj = objects[i];

        if ((int)obj.rect.width <= 0 || (int)obj.rect.height <= 0)
            continue;

        const ncnn::Mat mm = objects_mask.channel(i);

        obj.mask = cv::Mat((int)obj.rect.height, (int)obj.rect.width, CV_8UC1);

        for (int y = 0; y < (int)obj.rect.height; y++)
        {
            int src_y = (int)(hpad / 2 / scale + obj.rect.y + y);
            int src_x = (int)(wpad / 2 / scale + obj.rect.x);

            if (src_y < 0 || src_y >= objects_mask.h || src_x < 0)
                continue;
            const float* pmm = mm.row(src_y) + src_x;
            uchar* pmask = obj.mask.ptr<uchar>(y);
            for (int x = 0; x < (int)obj.rect.width; x++)
            {
                if (src_x + x < objects_mask.w)
                    pmask[x] = pmm[x] > mask_threshold ? 1 : 0;
                else
                    pmask[x] = 0;
            }
        }
    }

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

int YOLO26_seg::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    static cv::Scalar colors[] = {
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

        if (!obj.mask.empty())
        {
        for (int y = 0; y < (int)obj.rect.height; y++)
        {
            if ((int)obj.rect.y + y < 0 || (int)obj.rect.y + y >= rgb.rows)
                continue;
            const uchar* maskptr = obj.mask.ptr<const uchar>(y);
            uchar* bgrptr = rgb.ptr<uchar>((int)obj.rect.y + y) + (int)obj.rect.x * 3;
            for (int x = 0; x < (int)obj.rect.width; x++)
            {
                if (maskptr[x])
                {
                    bgrptr[0] = bgrptr[0] * 0.5 + color[0] * 0.5;
                    bgrptr[1] = bgrptr[1] * 0.5 + color[1] * 0.5;
                    bgrptr[2] = bgrptr[2] * 0.5 + color[2] * 0.5;
                }
                bgrptr += 3;
            }
        }
        }

        cv::rectangle(rgb, obj.rect, color);

        char text[256];
        const int num_class_names = sizeof(class_names) / sizeof(class_names[0]);
        if (obj.label >= 0 && obj.label < num_class_names)
            sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
        else
            sprintf(text, "class%d %.1f%%", obj.label, obj.prob * 100);

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
