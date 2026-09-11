#ifndef YOLO_FEATURE_EXTRACTOR_HPP
#define YOLO_FEATURE_EXTRACTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "DatasetLoader.hpp"
#include "FeatureExtractor.hpp" // For FeatureSample definition

class YoloFeatureExtractor {
public:
    YoloFeatureExtractor();

    // Extract feature vector from video sequence using YOLO tracked bounding boxes
    FeatureSample extractFromSequence(const std::vector<cv::Mat>& frames,
                                       int label,
                                       const std::string& sequence_name,
                                       const std::vector<cv::Rect>& yolo_boxes);
};

#endif // YOLO_FEATURE_EXTRACTOR_HPP
