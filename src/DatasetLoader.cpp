/**
AUTHOR: ROSSETTO LUCA
*/

#include "DatasetLoader.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

bool DatasetLoader::parseGroundTruth(const std::string& txt_path, int img_w, int img_h,
                                     int& label, cv::Rect& bbox) {
    std::ifstream in_file(txt_path);
    if (!in_file.is_open()) {
        std::cerr << "[ERROR] Cannot open txt file: " << txt_path << std::endl;
        return false;
    }

    float xc, yc, w, h;
    // Read format: <class_id> <x_center> <y_center> <width> <height>
    if (in_file >> label >> xc >> yc >> w >> h) {
        float abs_xc = (xc <= 1.0f) ? (xc * img_w) : xc;
        float abs_yc = (yc <= 1.0f) ? (yc * img_h) : yc;
        float abs_w  = (w <= 1.0f)  ? (w * img_w)  : w;
        float abs_h  = (h <= 1.0f)  ? (h * img_h)  : h;

        int x = (int)(abs_xc - abs_w / 2.0f);
        int y = (int)(abs_yc - abs_h / 2.0f);
        int width = (int)abs_w;
        int height = (int)abs_h;

        cv::Rect raw_box(x, y, width, height);
        cv::Rect img_box(0, 0, img_w, img_h);

        bbox = raw_box & img_box;
        return true;
    }
    return false;
}

std::vector<SequenceData> DatasetLoader::loadDataset(const std::string& dataset_path) {
    std::vector<SequenceData> dataset;

    if (!fs::exists(dataset_path)) {
        std::cerr << "[ERROR] Folder does not exist: " << dataset_path << std::endl;
        return dataset;
    }

    for (const auto& class_dir : fs::directory_iterator(dataset_path)) {
        if (!class_dir.is_directory()) continue;

        for (const auto& seq_dir : fs::directory_iterator(class_dir.path())) {
            if (!seq_dir.is_directory()) continue;

            SequenceData seq;
            seq.sequence_name = seq_dir.path().filename().string();

            fs::path frames_path = seq_dir.path() / "data";
            fs::path labels_path = seq_dir.path() / "labels";

            if (!fs::exists(frames_path)) frames_path = seq_dir.path();
            if (!fs::exists(labels_path)) labels_path = seq_dir.path();

            std::vector<std::string> frame_files;
            if (fs::exists(frames_path) && fs::is_directory(frames_path)) {
                for (const auto& f : fs::directory_iterator(frames_path)) {
                    std::string ext = f.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") {
                        frame_files.push_back(f.path().string());
                    }
                }
            }

            std::sort(frame_files.begin(), frame_files.end());

            for (const auto& f_path : frame_files) {
                cv::Mat frame_img = cv::imread(f_path);
                if (!frame_img.empty()) {
                    seq.frames.push_back(frame_img);
                }
            }

            std::string gt_file = "";
            if (fs::exists(labels_path) && fs::is_directory(labels_path)) {
                for (const auto& f : fs::directory_iterator(labels_path)) {
                    if (f.path().extension() == ".txt") {
                        gt_file = f.path().string();
                        break;
                    }
                }
            }

            if (seq.frames.empty() || gt_file.empty()) {
                continue;
            }

            int img_w = seq.frames[0].cols;
            int img_h = seq.frames[0].rows;
            if (parseGroundTruth(gt_file, img_w, img_h, seq.class_label, seq.median_bbox)) {
                dataset.push_back(seq);
            }
        }
    }

    return dataset;
}
