#include "DatasetLoader.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

bool DatasetLoader::parseGroundTruth(const std::string& txt_path, int img_width, int img_height,
                                     int& out_label, cv::Rect& out_bbox) {
    std::ifstream file(txt_path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Unable to open annotation file: " << txt_path << std::endl;
        return false;
    }

    float x_c, y_c, w, h;
    // Format: <class_id> <x_center> <y_center> <width> <height>
    if (file >> out_label >> x_c >> y_c >> w >> h) {
        // Auto-detect if coordinates are normalized [0.0, 1.0] or absolute pixels
        float abs_x_c = (x_c <= 1.0f) ? (x_c * img_width) : x_c;
        float abs_y_c = (y_c <= 1.0f) ? (y_c * img_height) : y_c;
        float abs_w   = (w <= 1.0f)   ? (w * img_width)   : w;
        float abs_h   = (h <= 1.0f)   ? (h * img_height)  : h;

        int x = static_cast<int>(abs_x_c - (abs_w / 2.0f));
        int y = static_cast<int>(abs_y_c - (abs_h / 2.0f));
        int width = static_cast<int>(abs_w);
        int height = static_cast<int>(abs_h);

        cv::Rect raw_bbox(x, y, width, height);
        cv::Rect img_bounds(0, 0, img_width, img_height);

        // Safe geometric intersection with frame limits
        out_bbox = raw_bbox & img_bounds;
        return true;
    }
    return false;
}

std::vector<SequenceData> DatasetLoader::loadDataset(const std::string& dataset_root_path) {
    std::vector<SequenceData> dataset;

    if (!fs::exists(dataset_root_path)) {
        std::cerr << "[ERROR] Dataset path does not exist: " << dataset_root_path << std::endl;
        return dataset;
    }

    for (const auto& action_entry : fs::directory_iterator(dataset_root_path)) {
        if (!action_entry.is_directory()) continue;

        for (const auto& seq_entry : fs::directory_iterator(action_entry.path())) {
            if (!seq_entry.is_directory()) continue;

            SequenceData seq;
            seq.sequence_name = seq_entry.path().filename().string();

            fs::path frames_dir = seq_entry.path() / "data";
            fs::path labels_dir = seq_entry.path() / "labels";

            if (!fs::exists(frames_dir)) frames_dir = seq_entry.path();
            if (!fs::exists(labels_dir)) labels_dir = seq_entry.path();

            std::vector<std::string> frame_paths;
            if (fs::exists(frames_dir) && fs::is_directory(frames_dir)) {
                for (const auto& file : fs::directory_iterator(frames_dir)) {
                    std::string ext = file.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                    if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") {
                        frame_paths.push_back(file.path().string());
                    }
                }
            }

            std::sort(frame_paths.begin(), frame_paths.end());

            for (const auto& path : frame_paths) {
                cv::Mat frame = cv::imread(path);
                if (!frame.empty()) {
                    seq.frames.push_back(frame);
                }
            }

            std::string gt_file_path = "";
            if (fs::exists(labels_dir) && fs::is_directory(labels_dir)) {
                for (const auto& file : fs::directory_iterator(labels_dir)) {
                    if (file.path().extension() == ".txt") {
                        gt_file_path = file.path().string();
                        break;
                    }
                }
            }

            if (seq.frames.empty() || gt_file_path.empty()) {
                std::cerr << "[WARNING] Missing frames or annotation in: " 
                          << action_entry.path().filename().string() << "/" << seq.sequence_name << std::endl;
                continue;
            }

            int img_w = seq.frames[0].cols;
            int img_h = seq.frames[0].rows;
            if (parseGroundTruth(gt_file_path, img_w, img_h, seq.class_label, seq.median_bbox)) {
                dataset.push_back(seq);
                std::cout << "[INFO] Loaded " << seq.sequence_name 
                          << " | Frames: " << seq.frames.size() 
                          << " | Label: " << seq.class_label << std::endl;
            }
        }
    }

    return dataset;
}
