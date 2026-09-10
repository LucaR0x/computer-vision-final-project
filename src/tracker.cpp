#include "tracker.hpp"
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <algorithm>
#include <cmath>

Tracker::Tracker() {
    bg_subtractor = cv::createBackgroundSubtractorKNN(200, 400.0, false);
    first_frame = true;
    bg_initialized = false;
}

void Tracker::init(const std::vector<cv::Mat>& sequence_frames) {
    if (sequence_frames.empty()) return;

    int rows = sequence_frames[0].rows;
    int cols = sequence_frames[0].cols;
    int num_frames = (int)sequence_frames.size();

    bg_median = cv::Mat::zeros(rows, cols, CV_8UC1);
    std::vector<uchar> pixel_values(num_frames);

    // Compute pixel-wise median across sequence frames for background estimation
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            for (int i = 0; i < num_frames; ++i) {
                cv::Mat gray_f;
                if (sequence_frames[i].channels() == 3) {
                    cv::cvtColor(sequence_frames[i], gray_f, cv::COLOR_BGR2GRAY);
                } else {
                    gray_f = sequence_frames[i];
                }
                pixel_values[i] = gray_f.at<uchar>(r, c);
            }
            std::nth_element(pixel_values.begin(), pixel_values.begin() + num_frames / 2, pixel_values.end());
            bg_median.at<uchar>(r, c) = pixel_values[num_frames / 2];
        }
    }

    bg_initialized = true;
}

cv::Rect Tracker::processFrame(const cv::Mat& frame, cv::Mat& out_mask) {
    cv::Mat gray_img;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray_img, cv::COLOR_BGR2GRAY);
    } else {
        gray_img = frame.clone();
    }

    int img_w = gray_img.cols;
    int img_h = gray_img.rows;

    cv::Mat fg_mask;
    if (bg_initialized && !bg_median.empty()) {
        cv::Mat diff_img;
        cv::absdiff(gray_img, bg_median, diff_img);
        cv::GaussianBlur(diff_img, diff_img, cv::Size(5, 5), 1.0);
        cv::threshold(diff_img, fg_mask, 18, 255, cv::THRESH_BINARY);

        // Fallback for low-contrast frames
        if (cv::countNonZero(fg_mask) < 35) {
            cv::threshold(diff_img, fg_mask, 10, 255, cv::THRESH_BINARY);
        }
    } else {
        cv::Mat fg_knn;
        double lr = first_frame ? 0.5 : 0.0002;
        bg_subtractor->apply(gray_img, fg_knn, lr);
        cv::threshold(fg_knn, fg_mask, 150, 255, cv::THRESH_BINARY);
    }

    // Clean image border noise
    cv::rectangle(fg_mask, cv::Rect(0, 0, img_w, img_h), cv::Scalar(0), 6);

    // Morphological closing using vertical kernel to connect head, torso and legs
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 25));
    cv::morphologyEx(fg_mask, out_mask, cv::MORPH_CLOSE, close_kernel);
    cv::dilate(out_mask, out_mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 9)));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(out_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect main_box(0, 0, 0, 0);
    double max_area = 0.0;

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        cv::Rect b = cv::boundingRect(contours[i]);
        if (b.width > img_w * 0.85 || b.height > img_h * 0.85) continue;

        if (area > max_area) {
            max_area = area;
            main_box = b;
        }
    }

    cv::Rect final_box = main_box;
    if (final_box.area() > 0) {
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area < 25.0) continue;
            cv::Rect b = cv::boundingRect(contours[i]);
            if (b == main_box) continue;
            if (b.width > img_w * 0.85 || b.height > img_h * 0.85) continue;

            int dx = std::max(0, std::max(main_box.x - (b.x + b.width), b.x - (main_box.x + main_box.width)));
            int dy = std::max(0, std::max(main_box.y - (b.y + b.height), b.y - (main_box.y + main_box.height)));
            if (dx < 20 && dy < 20) {
                final_box |= b;
            }
        }
    }

    // Aspect ratio human physical proportion check
    if (final_box.area() > 0) {
        float aspect_ratio = (float)final_box.height / ((float)final_box.width + 1e-4f);
        if (aspect_ratio < 1.4f) {
            int target_h = (int)(final_box.width * 2.0f);
            target_h = std::min(target_h, (int)(img_h * 0.65f));
            if (target_h > final_box.height) {
                int pad_y = (target_h - final_box.height) / 2;
                final_box.y = std::max(0, final_box.y - pad_y);
                final_box.height = std::min(img_h - final_box.y, target_h);
            }
        }
    }

    first_frame = false;
    return final_box;
}