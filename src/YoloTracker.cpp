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
    
    // Gestione sicura delle dimensioni indipendentemente dalla versione di OpenCV
    int rows = (out.dims == 3) ? out.size[1] : out.size[0];
    int cols = (out.dims == 3) ? out.size[2] : out.size[1];

    cv::Mat predictions(rows, cols, CV_32F, out.ptr<float>());
    predictions = predictions.clone().t(); // Il clone libera la memoria dal tensore originale

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
        
        best_bbox &= cv::Rect(0, 0, img_w, img_h);
        
        const auto& kpts = keypoints_list[idx];
        if (kpts.size() >= 3) {
            // Convertiamo i Point2f in Point (interi) PRIMA del convexHull
            std::vector<cv::Point> kpts_int;
            for (const auto& pt : kpts) {
                kpts_int.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
            }

            std::vector<cv::Point> hull;
            cv::convexHull(kpts_int, hull);
            
            // Disegniamo il poligono pieno
            cv::fillConvexPoly(out_mask, hull, cv::Scalar(255));
            
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
            cv::dilate(out_mask, out_mask, kernel);
        } else {
            cv::rectangle(out_mask, best_bbox, cv::Scalar(255), cv::FILLED);
        }

        prev_bbox = best_bbox;
    }

    return best_bbox;
}