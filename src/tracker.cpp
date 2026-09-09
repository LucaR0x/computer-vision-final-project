#include "tracker.hpp"

Tracker::Tracker() {
    bg_subtractor = cv::createBackgroundSubtractorKNN(100, 400.0, false);
    first_frame = true;
}

cv::Rect Tracker::processFrame(const cv::Mat& frame, cv::Mat& out_mask) {
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    int img_w = gray.cols;
    int img_h = gray.rows;

    cv::Mat fg_mask;
    double learning_rate = first_frame ? 0.3 : 0.001;
    bg_subtractor->apply(gray, fg_mask, learning_rate);

    // Threshold shadows
    cv::threshold(fg_mask, fg_mask, 180, 255, cv::THRESH_BINARY);

    // Filter out edge noise around image border
    cv::rectangle(fg_mask, cv::Rect(0, 0, img_w, img_h), cv::Scalar(0), 4);

    // Morphological open & close to isolate human body silhouette
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 15));
    
    cv::morphologyEx(fg_mask, out_mask, cv::MORPH_OPEN, kernel_open);
    cv::morphologyEx(out_mask, out_mask, cv::MORPH_CLOSE, kernel_close);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(out_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect current_bbox(0, 0, 0, 0);
    double max_area = 0;

    // Find largest valid human contour
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        cv::Rect b = cv::boundingRect(contour);
        // Exclude full-frame edge noise
        if (b.width > img_w * 0.85 || b.height > img_h * 0.85) continue;

        if (area > max_area && area > 60.0) {
            max_area = area;
            current_bbox = b;
        }
    }

    if (current_bbox.area() > 0) {
        if (first_frame) {
            prev_bbox = current_bbox;
            first_frame = false;
        } else {
            // Adaptive smoothing: higher alpha (0.8) to prevent lag during fast motion
            float alpha = 0.8f;
            int new_x = static_cast<int>(std::round(current_bbox.x * alpha + prev_bbox.x * (1.0f - alpha)));
            int new_y = static_cast<int>(std::round(current_bbox.y * alpha + prev_bbox.y * (1.0f - alpha)));
            int new_w = static_cast<int>(std::round(current_bbox.width * alpha + prev_bbox.width * (1.0f - alpha)));
            int new_h = static_cast<int>(std::round(current_bbox.height * alpha + prev_bbox.height * (1.0f - alpha)));

            current_bbox = cv::Rect(new_x, new_y, new_w, new_h);
            prev_bbox = current_bbox;
        }
    } else {
        current_bbox = prev_bbox;
    }

    return current_bbox;
}