

#include "yolo26.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <android/log.h>

static int g_last_inferred_num_class = 80;

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

static inline float calibrate_probability(float p)
{
    const float eps = 1e-6f;
    if (p < eps) p = eps;
    if (p > 1.f - eps) p = 1.f - eps;

    const float T = 6.0f;
    const float logit = logf(p / (1.f - p));
    float cal = 1.f / (1.f + expf(-(logit / T)));

    cal = powf(cal, 1.6f);
    return cal;
}

static inline bool is_valid_f32(float v)
{

    return v == v && v > -1e20f && v < 1e20f;
}

static void generate_proposals_yolo26(const ncnn::Mat& pred, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects, bool transposed = false)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int num_class = transposed ? (pred.w - 4) : (pred.h - 4);

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            int box_idx = y * num_grid_x + x;

            int label = -1;
            float score = -FLT_MAX;

            if (transposed) {

                const float* row_ptr = pred.row(box_idx);

                for (int k = 0; k < num_class; k++)
                {
                    float s = row_ptr[4 + k];
                    if (s > score)
                    {
                        label = k;
                        score = s;
                    }
                }

                float score_cal = calibrate_probability(score);
                if (score_cal >= prob_threshold)
                {

                    float pred_ltrb[4];
                    for (int k = 0; k < 4; k++)
                    {
                        pred_ltrb[k] = row_ptr[k] * stride;
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
                    obj.prob = score_cal;

                    objects.push_back(obj);
                }
            }
            else {

                for (int k = 0; k < num_class; k++)
                {
                    float s = pred.row(4 + k)[box_idx];
                    if (s > score)
                    {
                        label = k;
                        score = s;
                    }
                }

                float score_cal = calibrate_probability(score);
                if (score_cal >= prob_threshold)
                {

                    float pred_ltrb[4];
                    for (int k = 0; k < 4; k++)
                    {
                        pred_ltrb[k] = pred.row(k)[box_idx] * stride;
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
                    obj.prob = score_cal;

                    objects.push_back(obj);
                }
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
                ncnn::Mat pred_bbox = pred_grid.range(0, reg_max_1 * 4).reshape(reg_max_1, 4);

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

        generate_proposals(pred.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects);
        pred_row_offset += num_grid;
    }
}

static void generate_proposals_yolo26_transposed(const ncnn::Mat& pred, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int img_w = in_pad.w;
    const int img_h = in_pad.h;

    const int num_class = pred.h - 4;
    const int total_boxes = pred.w;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "YOLO26 transposed: num_class=%d, total_boxes=%d", num_class, total_boxes);

    int detected_count = 0;
    (void)strides;

    for (int box_idx = 0; box_idx < total_boxes; box_idx++)
    {

        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float s = pred.row(4 + k)[box_idx];
            if (s > score)
            {
                label = k;
                score = s;
            }
        }

        if (label < 0 || !is_valid_f32(score))
            continue;

        float display_score = score;
        {
            const float eps = 1e-6f;
            float p = std::max(eps, std::min(1.f - eps, score));
            float logit = logf(p / (1.f - p));
            display_score = 1.f / (1.f + expf(-(logit / 3.0f)));
        }

        if (display_score < prob_threshold)
            continue;

        float x_center = pred.row(0)[box_idx];
        float y_center = pred.row(1)[box_idx];
        float bw = pred.row(2)[box_idx];
        float bh = pred.row(3)[box_idx];

        if (!is_valid_f32(x_center) || !is_valid_f32(y_center) || !is_valid_f32(bw) || !is_valid_f32(bh))
            continue;
        if (bw <= 0.f || bh <= 0.f)
            continue;
        if (bw < 5.f || bh < 5.f)
            continue;
        const float aspect = bw > bh ? (bw / bh) : (bh / bw);
        if (aspect > 8.f)
            continue;
        if (bw > img_w * 0.95f || bh > img_h * 0.95f)
            continue;

        {
            float vis_x0 = std::max(0.f, x_center - bw * 0.5f);
            float vis_y0 = std::max(0.f, y_center - bh * 0.5f);
            float vis_x1 = std::min((float)img_w, x_center + bw * 0.5f);
            float vis_y1 = std::min((float)img_h, y_center + bh * 0.5f);
            float vis_area = std::max(0.f, vis_x1 - vis_x0) * std::max(0.f, vis_y1 - vis_y0);
            float full_area = bw * bh;
            if (full_area > 0.f && vis_area / full_area < 0.40f)
                continue;
        }

        float x0 = x_center - bw * 0.5f;
        float y0 = y_center - bh * 0.5f;

        if (detected_count < 10) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                "Det[%d] box=%d xywh=(%.1f,%.1f,%.1f,%.1f) label=%d raw=%.4f cal=%.4f",
                detected_count, box_idx, x_center, y_center, bw, bh, label, score, display_score);
        }
        detected_count++;

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = bw;
        obj.rect.height = bh;
        obj.label = label;
        obj.prob = display_score;
        objects.push_back(obj);

        if (num_class == 10)
        {
            const int LABEL_PERSON = 5;
            if (label != LABEL_PERSON)
            {
                const bool person_related_primary = (label == 0 || label == 1 || label == 2 || label == 3 || label == 4 || label == 7 || label == 8 || label == 9);
                if (person_related_primary)
                {
                    float s_person = pred.row(4 + LABEL_PERSON)[box_idx];
                    float eps2 = 1e-6f;
                    float p2 = std::max(eps2, std::min(1.f - eps2, s_person));
                    float logit2 = logf(p2 / (1.f - p2));
                    float cal_person = 1.f / (1.f + expf(-(logit2 / 3.0f)));

                    if (cal_person >= prob_threshold * 0.85f)
                    {
                        Object obj2;
                        obj2.rect.x = x0;
                        obj2.rect.y = y0;
                        obj2.rect.width = bw;
                        obj2.rect.height = bh;
                        obj2.label = LABEL_PERSON;
                        obj2.prob = cal_person;
                        objects.push_back(obj2);
                    }
                }
            }
        }
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Total detected: %d boxes, %zu proposals (incl multi-class)", detected_count, objects.size());
}

static void filter_person_dependent_classes(std::vector<Object>& objects, int num_class)
{

    if (num_class != 10 || objects.empty())
        return;

    const int LABEL_NO_HARDHAT     = 2;
    const int LABEL_NO_MASK        = 3;
    const int LABEL_NO_SAFETY_VEST = 4;
    const int LABEL_PERSON         = 5;

    const int LABEL_HARDHAT     = 0;
    const int LABEL_MASK        = 1;
    const int LABEL_SAFETY_VEST = 7;

    std::vector<cv::Rect_<float>> person_boxes;
    for (const auto& obj : objects)
    {
        if (obj.label == LABEL_PERSON)
            person_boxes.push_back(obj.rect);
    }

    std::vector<Object> kept;
    kept.reserve(objects.size());

    for (const auto& obj : objects)
    {
        bool is_negative = (obj.label == LABEL_NO_HARDHAT ||
                            obj.label == LABEL_NO_MASK ||
                            obj.label == LABEL_NO_SAFETY_VEST);
        bool is_ppe_positive = (obj.label == LABEL_HARDHAT ||
                                obj.label == LABEL_MASK ||
                                obj.label == LABEL_SAFETY_VEST);

        if (!is_negative && !is_ppe_positive)
        {
            kept.push_back(obj);
            continue;
        }

        if (person_boxes.empty())
        {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                "PersonFilter: suppress label=%d prob=%.3f (no Person in scene)",
                obj.label, obj.prob);
            continue;
        }

        float box_area = obj.rect.width * obj.rect.height;
        if (box_area <= 0.f)
            continue;

        bool associated = false;
        int best_person_idx = -1;
        float best_overlap = 0.f;
        for (size_t pi = 0; pi < person_boxes.size(); pi++)
        {
            const cv::Rect_<float>& pbox = person_boxes[pi];
            cv::Rect_<float> inter = obj.rect & pbox;
            float inter_area = inter.area();
            float overlap = inter_area / box_area;
            if (overlap > best_overlap)
            {
                best_overlap = overlap;
                best_person_idx = (int)pi;
            }

            const float assoc_thr = is_negative ? 0.20f : 0.15f;
            if (overlap >= assoc_thr)
            {
                associated = true;
            }
        }

        if (associated)
        {
            Object adjusted = obj;

            if (is_ppe_positive && best_person_idx >= 0)
            {
                const cv::Rect_<float>& p = person_boxes[(size_t)best_person_idx];
                adjusted.rect.x = adjusted.rect.x * 0.35f + p.x * 0.65f;
                adjusted.rect.y = adjusted.rect.y * 0.35f + p.y * 0.65f;
                adjusted.rect.width = adjusted.rect.width * 0.35f + p.width * 0.65f;
                adjusted.rect.height = adjusted.rect.height * 0.35f + p.height * 0.65f;
            }
            kept.push_back(adjusted);
        }
        else
        {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                "PersonFilter: suppress label=%d prob=%.3f (no overlapping Person)",
                obj.label, obj.prob);
        }
    }

    if (kept.size() != objects.size())
    {
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
            "PersonFilter: %zu -> %zu objects", objects.size(), kept.size());
        objects.swap(kept);
    }
}

static void process_e2e_output(const ncnn::Mat& pred, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int num_detections = pred.h;
    const int det_size = pred.w;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "E2E mode: num_detections=%d, det_size=%d", num_detections, det_size);

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

        objects.push_back(obj);
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "E2E mode: detected %zu objects above threshold %.2f", objects.size(), prob_threshold);
}

int YOLO26_det::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    const int target_size = det_target_size;
    const float prob_threshold = det_prob_threshold;
    float effective_prob_threshold = prob_threshold;
    const float nms_threshold = det_nms_threshold;
    int inferred_num_class = -1;

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

    {
        const auto& input_names = yolo26.input_names();
        const auto& output_names = yolo26.output_names();
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "model input names:");
        for (size_t i = 0; i < input_names.size(); i++) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "%s", input_names[i]);
        }
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "model output names:");
        for (size_t i = 0; i < output_names.size(); i++) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "%s", output_names[i]);
        }
    }

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Output shape: w=%d h=%d c=%d elempack=%d elembits=%d cstep=%zu", out.w, out.h, out.c, out.elempack, out.elembits(), out.cstep);

    if (out.c == 1 && out.w > 4 && out.h > 4)
    {
        int num_boxes = 0;
        int num_class = 0;
        std::vector<float> max_cls_scores;

        if (out.w >= out.h)
        {

            num_boxes = out.w;
            num_class = out.h - 4;
            max_cls_scores.reserve(num_boxes);
            for (int i = 0; i < num_boxes; i++)
            {
                float smax = -FLT_MAX;
                for (int c = 0; c < num_class; c++)
                {
                    float s = out.row(4 + c)[i];
                    if (s > smax)
                        smax = s;
                }
                max_cls_scores.push_back(smax);
            }
        }
        else
        {

            num_boxes = out.h;
            num_class = out.w - 4;
            max_cls_scores.reserve(num_boxes);
            for (int i = 0; i < num_boxes; i++)
            {
                const float* row_ptr = out.row(i);
                float smax = -FLT_MAX;
                for (int c = 0; c < num_class; c++)
                {
                    float s = row_ptr[4 + c];
                    if (s > smax)
                        smax = s;
                }
                max_cls_scores.push_back(smax);
            }
        }

        if (!max_cls_scores.empty())
        {
            std::sort(max_cls_scores.begin(), max_cls_scores.end());
            const int n = (int)max_cls_scores.size();
            const auto qv = [&](float q) -> float {
                int idx = (int)(q * (n - 1));
                if (idx < 0) idx = 0;
                if (idx >= n) idx = n - 1;
                return max_cls_scores[idx];
            };

            int gt090 = 0;
            int gt099 = 0;
            for (float v : max_cls_scores)
            {
                if (v > 0.90f) gt090++;
                if (v > 0.99f) gt099++;
            }

            const float frac_gt099 = (float)gt099 / (float)n;

            if (num_class != 10) {
                if (frac_gt099 > 0.25f)
                    effective_prob_threshold = std::max(prob_threshold, 0.50f);
                else if (frac_gt099 > 0.15f)
                    effective_prob_threshold = std::max(prob_threshold, 0.45f);
            }

            __android_log_print(
                ANDROID_LOG_DEBUG,
                "ncnn",
                "RawScoreStats boxes=%d classes=%d p50=%.4f p90=%.4f p99=%.4f frac_gt090=%.6f frac_gt099=%.6f eff_prob=%.3f",
                num_boxes,
                num_class,
                qv(0.50f),
                qv(0.90f),
                qv(0.99f),
                (float)gt090 / (float)n,
                frac_gt099,
                effective_prob_threshold);
        }
    }

    std::vector<Object> proposals;

    int feature_dim = std::min(out.w, out.h);
    int num_boxes = std::max(out.w, out.h);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Detected: feature_dim=%d, num_boxes=%d", feature_dim, num_boxes);

    if (feature_dim == 6 && num_boxes <= 300) {

        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 E2E processing (no NMS needed)");

        ncnn::Mat out_e2e = out;
        if (out.w != 6) {

            out_e2e = ncnn::Mat(out.h, out.w);
            for (int i = 0; i < out.h; i++) {
                for (int j = 0; j < out.w; j++) {
                    out_e2e.row(j)[i] = out.row(i)[j];
                }
            }
        }

        process_e2e_output(out_e2e, in_pad, effective_prob_threshold, proposals);

        {
            int max_label_e2e = -1;
            for (int i = 0; i < out_e2e.h; i++)
            {
                int cid = (int)out_e2e.row(i)[4];
                if (cid > max_label_e2e)
                    max_label_e2e = cid;
            }
            if (max_label_e2e >= 0)
                g_last_inferred_num_class = max_label_e2e + 1;
        }

        objects.resize(proposals.size());
        for (size_t i = 0; i < proposals.size(); i++)
        {
            objects[i] = proposals[i];

            float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
            float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }

        filter_person_dependent_classes(objects, g_last_inferred_num_class);

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
    else if ((feature_dim > 4 && feature_dim < 64) || feature_dim == 84 || out.h == 84) {

        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 processing (no DFL, transposed format, custom classes supported)");
        inferred_num_class = feature_dim - 4;
        generate_proposals_yolo26_transposed(out, strides, in_pad, effective_prob_threshold, proposals);
    } else {

        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Using YOLO26 (Legacy) processing (with DFL)");
        inferred_num_class = feature_dim - 64;
        generate_proposals(out, strides, in_pad, effective_prob_threshold, proposals);
    }

    if (inferred_num_class > 0)
        g_last_inferred_num_class = inferred_num_class;

    qsort_descent_inplace(proposals);

    const size_t proposals_before_trim = proposals.size();

    if (inferred_num_class == 10) {

        const int per_class_topk = 50;
        std::vector<int> class_count(inferred_num_class, 0);
        std::vector<Object> balanced_proposals;
        balanced_proposals.reserve(per_class_topk * inferred_num_class);

        for (size_t i = 0; i < proposals.size(); i++) {
            int cls = proposals[i].label;
            if (cls >= 0 && cls < inferred_num_class && class_count[cls] < per_class_topk) {
                balanced_proposals.push_back(proposals[i]);
                class_count[cls]++;
            }
        }
        proposals.swap(balanced_proposals);

        qsort_descent_inplace(proposals);

        __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
            "SafeHat per-class topk: Person=%d SafeVest=%d Machinery=%d Vehicle=%d Hardhat=%d total=%zu",
            class_count[5], class_count[7], class_count[8], class_count[9], class_count[0],
            proposals.size());
    } else {
        int pre_nms_topk = 300;
        if (proposals_before_trim > 6000)
            pre_nms_topk = 60;
        else if (proposals_before_trim > 3000)
            pre_nms_topk = 90;
        else if (proposals_before_trim > 1500)
            pre_nms_topk = 120;
        if ((int)proposals.size() > pre_nms_topk)
            proposals.resize(pre_nms_topk);
    }

    const int builtin_class_count = 80;

    const bool is_safehat = (inferred_num_class == 10);
    const bool nms_agnostic = !is_safehat && ((inferred_num_class > 0 && inferred_num_class != builtin_class_count) || proposals_before_trim > 1500);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Post-process info: inferred_classes=%d builtin_classes=%d pretrim=%zu proposals_now=%zu", inferred_num_class, builtin_class_count, proposals_before_trim, proposals.size());
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold, nms_agnostic);

    int count = picked.size();

    if (inferred_num_class == 10) {
        int person_after_nms = 0;
        float person_max_prob = 0;
        for (int i = 0; i < count; i++) {
            if (proposals[picked[i]].label == 5) {
                person_after_nms++;
                if (proposals[picked[i]].prob > person_max_prob)
                    person_max_prob = proposals[picked[i]].prob;
            }
        }
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
            "PERSON-TRACK: after NMS=%d (max_prob=%.3f) out of %d picked",
            person_after_nms, person_max_prob, count);
    }
    int max_det = 20;
    if (inferred_num_class == 10) {

        max_det = 120;
    } else if (proposals_before_trim > 6000) {
        max_det = 8;
    } else if (proposals_before_trim > 3000) {
        max_det = 10;
    } else if (proposals_before_trim > 1500) {
        max_det = 12;
    }
    if (count > max_det)
        count = max_det;
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "After NMS: proposals=%zu picked=%d agnostic=%d", proposals.size(), count, nms_agnostic ? 1 : 0);

    objects.clear();
    objects.reserve(count);
    const int class_cap_size = inferred_num_class > 0 ? inferred_num_class : 80;
    std::vector<int> per_class_kept(class_cap_size, 0);
    const int max_per_class = (inferred_num_class == 10) ? 5 : (proposals_before_trim > 3000 ? 2 : 4);
    for (int i = 0; i < count; i++)
    {
        Object obj = proposals[picked[i]];

        float x0 = (obj.rect.x - (wpad / 2)) / scale;
        float y0 = (obj.rect.y - (hpad / 2)) / scale;
        float x1 = (obj.rect.x + obj.rect.width - (wpad / 2)) / scale;
        float y1 = (obj.rect.y + obj.rect.height - (hpad / 2)) / scale;

        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        float bw = x1 - x0;
        float bh = y1 - y0;

        if (!is_valid_f32(bw) || !is_valid_f32(bh))
            continue;

        if (bw < 4.f || bh < 4.f)
            continue;
        const float area = bw * bh;
        float min_area = proposals_before_trim > 3000 ? 196.f : 100.f;

        if (inferred_num_class == 10 && (obj.label == 5 || obj.label == 7 || obj.label == 4))
            min_area = 64.f;
        if (area < min_area)
            continue;
        const float aspect = bw > bh ? (bw / bh) : (bh / bw);
        float max_aspect = proposals_before_trim > 3000 ? 5.f : 7.f;
        if (inferred_num_class == 10 && (obj.label == 5 || obj.label == 7 || obj.label == 4))
            max_aspect = 12.f;
        if (aspect > max_aspect)
            continue;

        if (inferred_num_class == 10 && (obj.label == 8 || obj.label == 9))
        {
            float aspect_hw = bh / std::max(bw, 1.f);
            if (aspect_hw > 2.0f)
                continue;
        }

        if (obj.label >= 0 && obj.label < class_cap_size)
        {
            if (per_class_kept[obj.label] >= max_per_class)
                continue;
            per_class_kept[obj.label]++;
        }

        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = bw;
        obj.rect.height = bh;

        objects.push_back(obj);
    }

    if (inferred_num_class == 10) {
        int person_after_clip = 0;
        for (const auto& o : objects) { if (o.label == 5) person_after_clip++; }
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
            "PERSON-TRACK: after clip+filter=%d out of %zu objects", person_after_clip, objects.size());
    }

    if (!objects.empty())
    {
        qsort_descent_inplace(objects);
        std::vector<int> picked_final;
        nms_sorted_bboxes(objects, picked_final, 0.45f, !(inferred_num_class == 10));

        std::vector<Object> filtered;
        if (inferred_num_class == 10)
        {

            const int final_cap = 18;
            const int max_per_class_final = 3;
            std::vector<int> kept_per_class(10, 0);
            filtered.reserve(std::min((int)picked_final.size(), final_cap));

            int best_person_idx = -1;
            float best_person_prob = -1.f;

            for (size_t i = 0; i < picked_final.size(); i++)
            {
                const Object& cand = objects[picked_final[i]];
                if (cand.label == 5 && cand.prob > best_person_prob)
                {
                    best_person_prob = cand.prob;
                    best_person_idx = (int)i;
                }

                if ((int)filtered.size() >= final_cap)
                    continue;
                if (cand.label < 0 || cand.label >= 10)
                    continue;
                if (kept_per_class[cand.label] >= max_per_class_final)
                    continue;

                filtered.push_back(cand);
                kept_per_class[cand.label]++;
            }

            bool has_person = false;
            for (size_t i = 0; i < filtered.size(); i++)
            {
                if (filtered[i].label == 5)
                {
                    has_person = true;
                    break;
                }
            }
            if (!has_person && best_person_idx >= 0)
            {
                const Object& person_obj = objects[picked_final[(size_t)best_person_idx]];
                if ((int)filtered.size() < final_cap)
                {
                    filtered.push_back(person_obj);
                }
                else
                {

                    int replace_j = -1;
                    float min_prob = FLT_MAX;
                    for (size_t j = 0; j < filtered.size(); j++)
                    {
                        if (filtered[j].label == 5)
                            continue;
                        if (filtered[j].prob < min_prob)
                        {
                            min_prob = filtered[j].prob;
                            replace_j = (int)j;
                        }
                    }
                    if (replace_j >= 0)
                        filtered[(size_t)replace_j] = person_obj;
                }
            }
        }
        else
        {
            const int final_cap = proposals_before_trim > 3000 ? 10 : 15;
            filtered.reserve(std::min((int)picked_final.size(), final_cap));
            for (size_t i = 0; i < picked_final.size() && (int)filtered.size() < final_cap; i++)
            {
                filtered.push_back(objects[picked_final[i]]);
            }
        }

        objects.swap(filtered);
    }

    if (inferred_num_class == 10 && objects.size() > 1)
    {

        auto calc_iou = [](const cv::Rect2f& a, const cv::Rect2f& b) -> float {
            float x1 = std::max(a.x, b.x);
            float y1 = std::max(a.y, b.y);
            float x2 = std::min(a.x + a.width, b.x + b.width);
            float y2 = std::min(a.y + a.height, b.y + b.height);
            float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
            float union_area = a.width * a.height + b.width * b.height - inter;
            return union_area > 0.f ? inter / union_area : 0.f;
        };

        std::vector<bool> suppress(objects.size(), false);
        for (size_t i = 0; i < objects.size(); i++)
        {
            if (suppress[i]) continue;
            for (size_t j = i + 1; j < objects.size(); j++)
            {
                if (suppress[j]) continue;
                float iou = calc_iou(objects[i].rect, objects[j].rect);
                if (iou < 0.30f) continue;

                int li = objects[i].label;
                int lj = objects[j].label;

                bool conflict = false;

                if ((li == 4 && lj == 7) || (li == 7 && lj == 4)) conflict = true;

                if ((li == 2 && lj == 0) || (li == 0 && lj == 2)) conflict = true;

                if ((li == 3 && lj == 1) || (li == 1 && lj == 3)) conflict = true;

                if (conflict)
                {

                    if (objects[i].prob < objects[j].prob)
                        suppress[i] = true;
                    else
                        suppress[j] = true;
                    __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                        "MutualExcl: obj[%zu](label=%d,p=%.3f) vs obj[%zu](label=%d,p=%.3f) iou=%.2f -> suppress[%zu]",
                        i, li, objects[i].prob, j, lj, objects[j].prob, iou,
                        suppress[i] ? i : j);
                }
            }
        }

        std::vector<Object> kept;
        for (size_t i = 0; i < objects.size(); i++)
        {
            if (!suppress[i])
                kept.push_back(objects[i]);
        }
        objects.swap(kept);
    }

    filter_person_dependent_classes(objects, inferred_num_class > 0 ? inferred_num_class : g_last_inferred_num_class);

    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Final kept objects: %zu", objects.size());

    {
        static const char* safehat_names[] = {"Hardhat","Mask","No-Hardhat","No-Mask","No-SafeVest","Person","SafeCone","SafeVest","Machinery","Vehicle"};
        for (size_t i = 0; i < objects.size() && i < 20; i++) {
            const char* name = (objects[i].label >= 0 && objects[i].label < 10) ? safehat_names[objects[i].label] : "?";
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                "  Final[%zu] label=%d(%s) prob=%.3f box=(%.0f,%.0f,%.0f,%.0f)",
                i, objects[i].label, name, objects[i].prob,
                objects[i].rect.x, objects[i].rect.y, objects[i].rect.width, objects[i].rect.height);
        }
    }

    return 0;
}

int YOLO26_det::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names_coco80[] = {
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

    static const char* class_names_safehat10[] = {
        "Hardhat", "Mask", "No-Hardhat", "No-Mask", "No-Safety Vest",
        "Person", "Safety Cone", "Safety Vest", "Machinery", "Vehicle"
    };

    const char** class_names = class_names_coco80;
    int class_count = (int)(sizeof(class_names_coco80) / sizeof(class_names_coco80[0]));
    if (g_last_inferred_num_class == 10)
    {
        class_names = class_names_safehat10;
        class_count = (int)(sizeof(class_names_safehat10) / sizeof(class_names_safehat10[0]));
    }

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

        cv::rectangle(rgb, obj.rect, color);

        char text[256];
        const float prob_show = std::max(0.f, std::min(obj.prob, 1.f));
        if (obj.label >= 0 && obj.label < class_count)
        {
            sprintf(text, "%s p=%.4f", class_names[obj.label], prob_show);
        }
        else
        {
            sprintf(text, "class_%d p=%.4f", obj.label, prob_show);
        }

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
