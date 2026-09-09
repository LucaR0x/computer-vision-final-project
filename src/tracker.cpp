#include "tracker.hpp"

Tracker::Tracker() {
    // Initialize KNN subtractor suitable for short sequences
    bg_subtractor = cv::createBackgroundSubtractorKNN(100, 400.0, true);
    first_frame = true;
}

cv::Rect Tracker::processFrame(const cv::Mat& frame, cv::Mat& out_mask) {
    cv::Mat fg_mask;
    
    // Apply background subtraction with a high learning rate (0.05) to adapt quickly
    bg_subtractor->apply(frame, fg_mask, 0.05);

    // Remove shadows (KNN marks shadows as 127)
    cv::threshold(fg_mask, fg_mask, 200, 255, cv::THRESH_BINARY);

    // Morphological operations to clean the mask
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(fg_mask, out_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(out_mask, out_mask, cv::MORPH_CLOSE, kernel);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(out_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect current_bbox(0, 0, 0, 0);
    double max_area = 0;

    // Extract the largest contour representing the human
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > max_area && area > 100.0) {
            max_area = area;
            current_bbox = cv::boundingRect(contour);
        }
    }

    // Exponential Moving Average (EMA) for bounding box stabilization
    if (current_bbox.area() > 0) {
        if (first_frame) {
            prev_bbox = current_bbox;
            first_frame = false;
        } else {
            float alpha = 0.7f; // Smoothing factor
            current_bbox.x = current_bbox.x * alpha + prev_bbox.x * (1.0f - alpha);
            current_bbox.y = current_bbox.y * alpha + prev_bbox.y * (1.0f - alpha);
            current_bbox.width = current_bbox.width * alpha + prev_bbox.width * (1.0f - alpha);
            current_bbox.height = current_bbox.height * alpha + prev_bbox.height * (1.0f - alpha);
            
            prev_bbox = current_bbox;
        }
    } else {
        // Fallback to previous bbox if the subject is temporarily lost
        current_bbox = prev_bbox;
    }

    return current_bbox;
}