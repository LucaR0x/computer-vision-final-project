#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/opencv.hpp>
#include "tracker.hpp"

#include <vector>
#include <string>

struct FeatureSample {
    int label;
    std::string sequence_name;
    std::vector<float> descriptors;
};

class FeatureExtractor {
public:
    FeatureExtractor();

    FeatureSample extractFromSequence(const std::vector<cv::Mat>& frames, 
                                     int label, 
                                     const std::string& name,
                                     const cv::Rect& roi = cv::Rect());

private:
    int countSignalPeaks(const std::vector<float>& signal, float threshold);
};

#endif // FEATURE_EXTRACTOR_HPP