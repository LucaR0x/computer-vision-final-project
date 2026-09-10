#ifndef YOLO_TRACKER_HPP
#define YOLO_TRACKER_HPP


#include <opencv2/imgproc.hpp>
//#include <opencv2/geometry.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

class YoloTracker {
private:
    cv::dnn::Net net;
    cv::Rect prev_bbox;
    bool first_frame;
    
    const float SCORE_THRESHOLD = 0.5f;
    const float NMS_THRESHOLD = 0.4f;
    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;

public:
    YoloTracker(const std::string& model_path);
    
    // Ripristina lo stato del tracker per una nuova sequenza
    void reset();
    
    cv::Rect processFrame(const cv::Mat& frame, cv::Mat& out_mask);
};

#endif // YOLO_TRACKER_HPP