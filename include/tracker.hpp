#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <vector>

class Tracker {
private:
    cv::Ptr<cv::BackgroundSubtractor> bg_subtractor;
    cv::Rect prev_bbox;
    bool first_frame;

public:
    Tracker();
    
    // Processes the frame, returns the bounding box, and outputs the clean mask
    cv::Rect processFrame(const cv::Mat& frame, cv::Mat& out_mask);
};

#endif