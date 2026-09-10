#ifndef DATASET_LOADER_HPP
#define DATASET_LOADER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Structure to store a single action sequence
struct SequenceData {
    std::string sequence_name;
    int class_label;
    cv::Rect median_bbox;
    std::vector<cv::Mat> frames;
};

class DatasetLoader {
public:
    // Load all KTH action sequences from dataset folder
    static std::vector<SequenceData> loadDataset(const std::string& dataset_path);

private:
    // Parse ground truth txt annotation file
    static bool parseGroundTruth(const std::string& txt_path, int img_w, int img_h,
                                 int& label, cv::Rect& bbox);
};

#endif // DATASET_LOADER_HPP
