#include <iostream>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "tracker.hpp"

int main() {
    // Replace with the actual path to one of the dataset sequence folders
    std::string sequence_path = "/Users/lucarox/Desktop/CV_final_project/Sequences/walking/person01_walking_d1/data/"; 
    Tracker tracker;

    const int NUM_FRAMES = 40;
    const int MEDIAN_FRAME_IDX = 20; // 20th frame 

    cv::Rect median_bbox;

    for (int i = 1; i <= NUM_FRAMES; ++i) {
        // Load frame (adjust filename formatting based on the dataset structure)
        std::stringstream ss;
        ss << sequence_path << "frame_" << std::setfill('0') << std::setw(2) << i << ".png";
        cv::Mat frame = cv::imread(ss.str(), cv::IMREAD_GRAYSCALE);

        if (frame.empty()) {
            std::cerr << "Error: Could not load frame " << i << std::endl;
            continue;
        }

        cv::Mat clean_mask;
        cv::Rect bbox = tracker.processFrame(frame, clean_mask);

        // Save the 20th frame bbox for IoU evaluation
        if (i == MEDIAN_FRAME_IDX) {
            median_bbox = bbox;
            std::cout << "Median Frame (20th) BBox: [" 
                      << bbox.x << ", " << bbox.y << ", " 
                      << bbox.width << ", " << bbox.height << "]" << std::endl;
        }

        // --- Visualization ---
        cv::Mat display;
        cv::cvtColor(frame, display, cv::COLOR_GRAY2BGR);
        if (bbox.area() > 0) {
            cv::rectangle(display, bbox, cv::Scalar(0, 255, 0), 2);
        }
        
        cv::imshow("Tracking", display);
        cv::imshow("Segmentation Mask", clean_mask);
        if (i == 1) cv::waitKey(300);
        if (cv::waitKey(200) == 27) break; // Press ESC to exit early
    }

    cv::destroyAllWindows();
    return 0;
}