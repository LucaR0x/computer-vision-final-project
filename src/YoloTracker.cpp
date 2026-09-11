/**
AUTHOR: ROSSETTO LUCA
*/

#include "YoloTracker.hpp"
#include <iostream>

YoloTracker::YoloTracker(const std::string& model_path) {
    try {
        net = cv::dnn::readNetFromONNX(model_path);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Could not load YOLO model: " << e.what() << std::endl;
    }
    reset();
}

void YoloTracker::reset() {
    first_frame = true;
    prev_bbox = cv::Rect(0, 0, 0, 0);
}

cv::Rect YoloTracker::processFrame(const cv::Mat& frame, cv::Mat& out_mask) {
    if (frame.empty()) return prev_bbox;

    cv::Mat bgr_frame;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, bgr_frame, cv::COLOR_GRAY2BGR);
    } else {
        bgr_frame = frame.clone();
    }

    int img_w = bgr_frame.cols;
    int img_h = bgr_frame.rows;

    cv::Mat blob;
    cv::dnn::blobFromImage(bgr_frame, blob, 1.0 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    cv::Mat out = outputs[0];
    
    int rows = (out.dims == 3) ? out.size[1] : out.size[0];
    int cols = (out.dims == 3) ? out.size[2] : out.size[1];

    cv::Mat predictions(rows, cols, CV_32F, out.ptr<float>());
    predictions = predictions.clone().t();

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<std::vector<cv::Point2f>> keypoints_list;

    float x_factor = static_cast<float>(img_w) / INPUT_WIDTH;
    float y_factor = static_cast<float>(img_h) / INPUT_HEIGHT;

    for (int i = 0; i < predictions.rows; ++i) {
        float* data = predictions.ptr<float>(i);
        float confidence = data[4]; 

        if (confidence > SCORE_THRESHOLD) {
            scores.push_back(confidence);

            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            int left = static_cast<int>((cx - 0.5f * w) * x_factor);
            int top = static_cast<int>((cy - 0.5f * h) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);
            
            boxes.push_back(cv::Rect(left, top, width, height));

            std::vector<cv::Point2f> kpts;
            for (int k = 0; k < 17; ++k) {
                float kx = data[5 + k * 3];
                float ky = data[5 + k * 3 + 1];
                float k_conf = data[5 + k * 3 + 2];

                if (k_conf > 0.3f) { 
                    kpts.push_back(cv::Point2f(kx * x_factor, ky * y_factor));
                }
            }
            keypoints_list.push_back(kpts);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    out_mask = cv::Mat::zeros(img_h, img_w, CV_8UC1);
    cv::Rect best_bbox = prev_bbox;

    if (!indices.empty()) {
        int idx = indices[0]; 
        best_bbox = boxes[idx];
        
        const auto& kpts = keypoints_list[idx];
        if (kpts.size() >= 3) {
            std::vector<cv::Point> kpts_int;
            int min_x = img_w, min_y = img_h, max_x = 0, max_y = 0;
            
            for (const auto& pt : kpts) {
                int px = static_cast<int>(pt.x);
                int py = static_cast<int>(pt.y);
                kpts_int.push_back(cv::Point(px, py));
                
                if (px < min_x) min_x = px;
                if (py < min_y) min_y = py;
                if (px > max_x) max_x = px;
                if (py > max_y) max_y = py;
            }
            
            // Expand bounding box to enclose all keypoints
            cv::Rect kpt_rect(min_x, min_y, max_x - min_x, max_y - min_y);
            best_bbox |= kpt_rect;

            // Generate convex hull mask
            std::vector<cv::Point> hull;
            cv::convexHull(kpts_int, hull);
            cv::fillConvexPoly(out_mask, hull, cv::Scalar(255));
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(19, 19));
            cv::dilate(out_mask, out_mask, kernel);
        } else {
            cv::rectangle(out_mask, best_bbox, cv::Scalar(255), cv::FILLED);
        }

        // Apply dynamic padding
        int pad_x = static_cast<int>(best_bbox.width * 0.06);
        int pad_y = static_cast<int>(best_bbox.height * 0.02);
        best_bbox.x = std::max(0, best_bbox.x - pad_x);
        best_bbox.y = std::max(0, best_bbox.y - pad_y);
        best_bbox.width = std::min(img_w - best_bbox.x, best_bbox.width + 2 * pad_x);
        best_bbox.height = std::min(img_h - best_bbox.y, best_bbox.height + 2 * pad_y);
        best_bbox &= cv::Rect(0, 0, img_w, img_h);

        // Exponential moving average smoothing against jitter
        if (first_frame) {
            prev_bbox = best_bbox;
            first_frame = false;
        } else {
            float alpha = 0.80f; 
            int new_x = static_cast<int>(std::round(best_bbox.x * alpha + prev_bbox.x * (1.0f - alpha)));
            int new_y = static_cast<int>(std::round(best_bbox.y * alpha + prev_bbox.y * (1.0f - alpha)));
            int new_w = static_cast<int>(std::round(best_bbox.width * alpha + prev_bbox.width * (1.0f - alpha)));
            int new_h = static_cast<int>(std::round(best_bbox.height * alpha + prev_bbox.height * (1.0f - alpha)));

            best_bbox = cv::Rect(new_x, new_y, new_w, new_h);
            prev_bbox = best_bbox;
        }
    } else {
        // Fallback to previous bounding box if detection missed
        best_bbox = prev_bbox;
    }
    return best_bbox;
}