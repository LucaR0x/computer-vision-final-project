#ifndef DATASET_LOADER_HPP
#define DATASET_LOADER_HPP

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct SequenceData {
    std::string sequence_name;
    int class_label;
    cv::Rect median_bbox;
    std::vector<cv::Mat> frames;
};

class DatasetLoader {
public:
    static std::vector<SequenceData> loadDataset(const std::string& dataset_root_path);

private:
    static bool parseGroundTruth(const std::string& txt_path, int img_width, int img_height,
                                int& out_label, cv::Rect& out_bbox);
};

#endif // DATASET_LOADER_HPP
