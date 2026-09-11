#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "DatasetLoader.hpp"
#include "tracker.hpp"

// Structure to store feature vector for an action sequence
struct FeatureSample {
    std::string sequence_name;
    int label;
    std::vector<float> descriptors;
};

class FeatureExtractor {
public:
    FeatureExtractor();

    // Extract feature vector from video sequence using classical background subtractor tracking
    FeatureSample extractFromSequence(const std::vector<cv::Mat>& frames, 
                                       int label, 
                                       const std::string& sequence_name,
                                       const cv::Rect& ground_truth_roi);
};

#endif // FEATURE_EXTRACTOR_HPP