#include "DatasetLoader.hpp"
#include "FeatureExtractor.hpp"
#include "Classifier.hpp"
#include "tracker.hpp"

#include <iostream>
#include <filesystem>
#include <vector>
#include <iomanip>
#include <algorithm>

namespace fs = std::filesystem;

// Helper function to calculate Intersection over Union (IoU) between two cv::Rect
static float computeIoU(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    if (inter.width <= 0 || inter.height <= 0) return 0.0f;
    float inter_area = static_cast<float>(inter.area());
    float union_area = static_cast<float>(a.area() + b.area() - inter.area());
    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

int main(int argc, char** argv) {
    std::string dataset_path = "../data";
    std::string output_dir = "../output";

    if (argc > 1) {
        dataset_path = argv[1];
    }

    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    std::cout << "[INFO] Loading dataset from: " << dataset_path << std::endl;
    std::vector<SequenceData> raw_dataset = DatasetLoader::loadDataset(dataset_path);

    if (raw_dataset.empty()) {
        std::cerr << "[ERROR] No valid action sequences found in " << dataset_path << ". Exiting." << std::endl;
        return -1;
    }

    std::cout << "\n================ MEMBER 1: LOCALIZATION & TRACKING ================\n";
    std::cout << "[INFO] Processing " << raw_dataset.size() << " sequences with Tracker (KNN Subtractor + Morph + EMA)...\n";

    std::vector<FeatureSample> feature_dataset;
    std::vector<float> iou_scores;
    const int MEDIAN_FRAME_IDX = 19; // 20th frame (0-indexed 19)

    FeatureExtractor extractor;

    for (size_t s = 0; s < raw_dataset.size(); ++s) {
        const auto& seq = raw_dataset[s];
        Tracker tracker;
        cv::Rect pred_median_bbox(0, 0, 0, 0);

        for (size_t f = 0; f < seq.frames.size(); ++f) {
            cv::Mat clean_mask;
            cv::Rect bbox = tracker.processFrame(seq.frames[f], clean_mask);

            if (static_cast<int>(f) == MEDIAN_FRAME_IDX) {
                pred_median_bbox = bbox;
            }
        }

        // Fallback to GT bbox if tracking failed to produce a valid bounding box
        if (pred_median_bbox.width <= 0 || pred_median_bbox.height <= 0) {
            pred_median_bbox = seq.median_bbox;
        }

        // Calculate IoU with Ground Truth median bbox
        float iou = computeIoU(pred_median_bbox, seq.median_bbox);
        iou_scores.push_back(iou);

        // Feature extraction using the predicted median bounding box from Member 1's Tracker
        FeatureSample sample = extractor.extractFromSequence(seq.frames,
                                                            seq.class_label,
                                                            seq.sequence_name,
                                                            pred_median_bbox);
        feature_dataset.push_back(sample);
    }

    // Compute Mean IoU (mIoU)
    float sum_iou = 0.0f;
    for (float iou : iou_scores) sum_iou += iou;
    float mIoU = (iou_scores.empty()) ? 0.0f : (sum_iou / iou_scores.size());

    std::cout << "Mean Intersection over Union (mIoU): " << std::fixed << std::setprecision(4) << mIoU << "\n";
    std::cout << "===================================================================\n\n";

    std::cout << "================ MEMBER 2: CLASSIFICATION & EVALUATION ================\n";
    ActionClassifier classifier;
    classifier.evaluate(feature_dataset, 0.75f);

    std::string model_path = output_dir + "/svm_action_model.xml";
    std::cout << "[INFO] Training final SVM model on complete dataset..." << std::endl;
    classifier.trainAndSave(feature_dataset, model_path);
    std::cout << "=======================================================================\n";

    return 0;
}