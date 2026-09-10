#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/geometry.hpp>
#include <vector>

// Member 1: Tracker class for actor localization
class Tracker {
private:
    cv::Ptr<cv::BackgroundSubtractor> bg_subtractor;
    cv::Mat bg_median;
    bool bg_initialized;
    bool first_frame;

public:
    Tracker();
    
    // Calculate sequence median background image
    void init(const std::vector<cv::Mat>& sequence_frames);

    // Process a single frame and output bounding box and binary mask
    cv::Rect processFrame(const cv::Mat& frame, cv::Mat& out_mask);
};

#endif // TRACKER_HPP