#include "tracker.hpp"
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <algorithm>
#include <cmath>

Tracker::Tracker() {
    bg_subtractor = cv::createBackgroundSubtractorKNN(200, 400.0, false);
    first_frame = true;
    bg_initialized = false;
    prev_bbox = cv::Rect(0, 0, 0, 0);
}

void Tracker::init(const std::vector<cv::Mat>& sequence_frames) {
    if (sequence_frames.empty()) return;

    int h = sequence_frames[0].rows;
    int w = sequence_frames[0].cols;
    size_t num_frames = sequence_frames.size();

    bg_median = cv::Mat::zeros(h, w, CV_8UC1);
    std::vector<uchar> pixel_vals(num_frames);

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            for (size_t i = 0; i < num_frames; ++i) {
                cv::Mat g;
                if (sequence_frames[i].channels() == 3) {
                    cv::cvtColor(sequence_frames[i], g, cv::COLOR_BGR2GRAY);
                } else {
                    g = sequence_frames[i];
                }
                pixel_vals[i] = g.at<uchar>(r, c);
            }
            std::nth_element(pixel_vals.begin(), pixel_vals.begin() + num_frames / 2, pixel_vals.end());
            bg_median.at<uchar>(r, c) = pixel_vals[num_frames / 2];
        }
    }

    bg_initialized = true;
}

cv::Rect Tracker::processFrame(const cv::Mat& frame, cv::Mat& out_mask) {
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame.clone();
    }

    int img_w = gray.cols;
    int img_h = gray.rows;

    cv::Mat fg;
    if (bg_initialized && !bg_median.empty()) {
        cv::Mat diff;
        cv::absdiff(gray, bg_median, diff);
        cv::GaussianBlur(diff, diff, cv::Size(5, 5), 1.0);
        cv::threshold(diff, fg, 18, 255, cv::THRESH_BINARY);
    } else {
        cv::Mat fg_knn;
        double learning_rate = first_frame ? 0.5 : 0.0002;
        bg_subtractor->apply(gray, fg_knn, learning_rate);
        cv::threshold(fg_knn, fg, 150, 255, cv::THRESH_BINARY);
    }

    // Suppress border noise
    cv::rectangle(fg, cv::Rect(0, 0, img_w, img_h), cv::Scalar(0), 6);

    // Morphological closing with vertical kernel to bridge head-torso-legs
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 25));
    cv::morphologyEx(fg, out_mask, cv::MORPH_CLOSE, kernel_close);
    cv::dilate(out_mask, out_mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 9)));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(out_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect main_bbox(0, 0, 0, 0);
    double max_area = 0.0;

    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        cv::Rect b = cv::boundingRect(c);
        if (b.width > img_w * 0.85 || b.height > img_h * 0.85) continue;

        if (area > max_area) {
            max_area = area;
            main_bbox = b;
        }
    }

    cv::Rect bbox = main_bbox;
    if (bbox.area() > 0) {
        for (const auto& c : contours) {
            double area = cv::contourArea(c);
            if (area < 25.0) continue;
            cv::Rect b = cv::boundingRect(c);
            if (b == main_bbox) continue;
            if (b.width > img_w * 0.85 || b.height > img_h * 0.85) continue;

            int dist_x = std::max(0, std::max(main_bbox.x - (b.x + b.width), b.x - (main_bbox.x + main_bbox.width)));
            int dist_y = std::max(0, std::max(main_bbox.y - (b.y + b.height), b.y - (main_bbox.y + main_bbox.height)));
            if (dist_x < 20 && dist_y < 20) {
                bbox |= b;
            }
        }
    }

    if (bbox.area() > 0) {
        // Enforce human physical proportions (height/width ratio ~ 1.8 - 2.8)
        float aspect = static_cast<float>(bbox.height) / static_cast<float>(bbox.width + 1e-4f);
        if (aspect < 1.4f) {
            int target_h = static_cast<int>(bbox.width * 2.0f);
            target_h = std::min(target_h, static_cast<int>(img_h * 0.65f));
            if (target_h > bbox.height) {
                int pad_y = (target_h - bbox.height) / 2;
                bbox.y = std::max(0, bbox.y - pad_y);
                bbox.height = std::min(img_h - bbox.y, target_h);
            }
        }
    }

    first_frame = false;
    return bbox;
}