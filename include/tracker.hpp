#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/geometry.hpp>
#include <vector>

class Tracker {
private:
    cv::Ptr<cv::BackgroundSubtractor> bg_subtractor;
    cv::Mat bg_median;
    cv::Rect prev_bbox;
    bool bg_initialized;
    bool first_frame;

public:
    Tracker();
    
    // Initialize sequence background model
    void init(const std::vector<cv::Mat>& sequence_frames);

    // Processes the frame, returns the bounding box, and outputs the clean mask
    cv::Rect processFrame(const cv::Mat& frame, cv::Mat& out_mask);
};

#endif