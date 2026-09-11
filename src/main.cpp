#include "DatasetLoader.hpp"
#include "FeatureExtractor.hpp"
#include "YoloFeatureExtractor.hpp"
#include "Classifier.hpp"
#include "tracker.hpp"
#include "YoloTracker.hpp"

#include <iostream>
#include <filesystem>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <memory>

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

struct PipelineResults {
    std::string mode_name;
    float mIoU = 0.0f;
    float cv_accuracy = 0.0f;
    std::vector<float> per_class_miou;
};

static PipelineResults runPipeline(const std::vector<SequenceData>& raw_dataset,
                                  bool is_yolo_mode,
                                  const std::string& output_dir) {
    PipelineResults results;
    results.mode_name = is_yolo_mode ? "DEEP LEARNING (YOLOv8 Pose)" : "CLASSICAL COMPUTER VISION (KNN Subtractor)";

    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    std::cout << "\n===================================================================\n";
    std::cout << " PIPELINE: " << results.mode_name << "\n";
    std::cout << "===================================================================\n";

    std::cout << "\n================ MEMBER 1: LOCALIZATION & TRACKING ================\n";
    
    std::unique_ptr<YoloTracker> yolo_tracker = nullptr;
    if (is_yolo_mode) {
        std::cout << "[INFO] Mode: DEEP LEARNING FALLBACK (YOLOv8 Pose)\n";
        std::string model_path = "../models/yolov8n-pose.onnx";
        yolo_tracker = std::make_unique<YoloTracker>(model_path);
    } else {
        std::cout << "[INFO] Mode: CLASSICAL COMPUTER VISION (KNN Subtractor + Morph)\n";
    }

    std::cout << "[INFO] Processing " << raw_dataset.size() << " sequences...\n";

    std::vector<FeatureSample> feature_dataset;
    std::vector<float> iou_scores;
    std::vector<cv::Rect> pred_bboxes;
    const int MEDIAN_FRAME_IDX = 19; // 20th frame (0-indexed 19)

    FeatureExtractor cv_extractor;
    YoloFeatureExtractor yolo_extractor;

    for (size_t s = 0; s < raw_dataset.size(); ++s) {
        const auto& seq = raw_dataset[s];
        cv::Rect pred_median_bbox(0, 0, 0, 0);
        std::vector<cv::Rect> sequence_tracked_boxes;
        
        std::unique_ptr<Tracker> std_tracker = nullptr;

        if (is_yolo_mode) {
            yolo_tracker->reset();
        } else {
            std_tracker = std::make_unique<Tracker>();
            std_tracker->init(seq.frames);
        }

        for (size_t f = 0; f < seq.frames.size(); ++f) {
            cv::Mat clean_mask;
            cv::Rect bbox;
            
            if (is_yolo_mode) {
                bbox = yolo_tracker->processFrame(seq.frames[f], clean_mask);
            } else {
                bbox = std_tracker->processFrame(seq.frames[f], clean_mask);
            }

            sequence_tracked_boxes.push_back(bbox);

            if (static_cast<int>(f) == MEDIAN_FRAME_IDX) {
                pred_median_bbox = bbox;
            }
        }

        if (pred_median_bbox.width <= 0 || pred_median_bbox.height <= 0) {
            pred_median_bbox = seq.median_bbox;
        }

        pred_bboxes.push_back(pred_median_bbox);

        float iou = computeIoU(pred_median_bbox, seq.median_bbox);
        iou_scores.push_back(iou);

        FeatureSample sample;
        if (is_yolo_mode) {
            sample = yolo_extractor.extractFromSequence(seq.frames,
                                                        seq.class_label,
                                                        seq.sequence_name,
                                                        sequence_tracked_boxes);
        } else {
            sample = cv_extractor.extractFromSequence(seq.frames,
                                                      seq.class_label,
                                                      seq.sequence_name,
                                                      pred_median_bbox);
        }
        feature_dataset.push_back(sample);
    }

    // Compute Mean IoU (mIoU) per class and overall
    float sum_iou = 0.0f;
    std::vector<float> class_iou_sums(6, 0.0f);
    std::vector<int> class_iou_counts(6, 0);

    for (size_t i = 0; i < raw_dataset.size(); ++i) {
        int lbl = raw_dataset[i].class_label;
        if (lbl >= 1 && lbl <= 6) {
            class_iou_sums[lbl - 1] += iou_scores[i];
            class_iou_counts[lbl - 1]++;
        }
        sum_iou += iou_scores[i];
    }
    results.mIoU = (iou_scores.empty()) ? 0.0f : (sum_iou / iou_scores.size());

    results.per_class_miou.resize(6, 0.0f);
    const std::string class_names_arr[6] = {"Boxing", "Clapping", "Waving", "Jogging", "Running", "Walking"};
    std::cout << "Mean Intersection over Union (mIoU): " << std::fixed << std::setprecision(4) << results.mIoU << "\n";
    std::cout << "Per-class mIoU breakdown:\n";
    for (int c = 0; c < 6; ++c) {
        float c_miou = (class_iou_counts[c] > 0) ? (class_iou_sums[c] / class_iou_counts[c]) : 0.0f;
        results.per_class_miou[c] = c_miou;
        std::cout << "  - " << std::setw(10) << class_names_arr[c] << ": " << std::setprecision(4) << c_miou << "\n";
    }
    std::cout << "===================================================================\n\n";

    std::cout << "================ MEMBER 2: CLASSIFICATION & EVALUATION ================\n";
    ActionClassifier classifier;
    results.cv_accuracy = classifier.evaluate(feature_dataset, 0.75f);

    std::string model_path = output_dir + "/svm_action_model.xml";
    std::cout << "[INFO] Training final SVM model on complete dataset..." << std::endl;
    classifier.trainAndSave(feature_dataset, model_path);

    // Save bounding box visual outputs for all 72 sequences
    std::string viz_dir = output_dir + "/visualizations";
    if (!fs::exists(viz_dir)) {
        fs::create_directories(viz_dir);
    }

    std::cout << "\n[INFO] Saving bounding box visualization images to: " << viz_dir << std::endl;
    for (size_t s = 0; s < raw_dataset.size(); ++s) {
        const auto& seq = raw_dataset[s];
        int pred_class = classifier.predict(feature_dataset[s].descriptors);
        int gt_class = seq.class_label;

        cv::Mat viz_img;
        if (seq.frames[MEDIAN_FRAME_IDX].channels() == 1) {
            cv::cvtColor(seq.frames[MEDIAN_FRAME_IDX], viz_img, cv::COLOR_GRAY2BGR);
        } else {
            viz_img = seq.frames[MEDIAN_FRAME_IDX].clone();
        }

        // Draw Ground Truth bounding box in BLUE
        cv::rectangle(viz_img, seq.median_bbox, cv::Scalar(255, 0, 0), 2);

        // Draw Predicted bounding box from Member 1 Tracker in GREEN
        cv::rectangle(viz_img, pred_bboxes[s], cv::Scalar(0, 255, 0), 2);

        std::string gt_str = (gt_class >= 1 && gt_class <= 6) ? class_names_arr[gt_class - 1] : "Unknown";
        std::string pred_str = (pred_class >= 1 && pred_class <= 6) ? class_names_arr[pred_class - 1] : "Unknown";

        std::string text_line = "GT:" + gt_str + " | PRED:" + pred_str + " | IoU:" + std::to_string(iou_scores[s]).substr(0, 4);
        cv::putText(viz_img, text_line, cv::Point(5, 12), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 255, 255), 1);

        std::string out_path = viz_dir + "/" + seq.sequence_name + ".png";
        cv::imwrite(out_path, viz_img);
    }
    std::cout << "[INFO] Saved 72 visualization images into " << viz_dir << std::endl;
    std::cout << "=======================================================================\n";

    return results;
}

int main(int argc, char** argv) {
    std::string dataset_path = "../data";
    std::string output_dir = "../output";
    bool use_yolo = false;
    bool use_all = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use-yolo") {
            use_yolo = true;
        } else if (arg == "--use-all") {
            use_all = true;
        } else if (arg.rfind("--", 0) != 0) { // If it doesn't start with "--", assume it's the dataset path
            dataset_path = arg;
        }
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

    if (use_all) {
        // Run both Classical CV and Deep Learning YOLO pipelines and compare them
        std::cout << "\n=======================================================================\n";
        std::cout << "               RUNNING COMPLETE COMPARISON (--use-all)                 \n";
        std::cout << "=======================================================================\n";

        PipelineResults cv_res = runPipeline(raw_dataset, false, output_dir + "/cv");
        PipelineResults yolo_res = runPipeline(raw_dataset, true, output_dir + "/yolo");

        const std::string class_names_arr[6] = {"Boxing", "Clapping", "Waving", "Jogging", "Running", "Walking"};

        std::cout << "\n\n";
        std::cout << "================================================================================\n";
        std::cout << "                      FINAL COMPARISON & EVALUATION SUMMARY                      \n";
        std::cout << "================================================================================\n";
        std::cout << std::left << std::setw(30) << " Metric / Class"
                  << std::setw(25) << " Classical Computer Vision"
                  << std::setw(25) << " Deep Learning (YOLOv8 Pose)"
                  << "\n";
        std::cout << std::string(80, '-') << "\n";

        std::cout << std::left << std::setw(30) << " Mean IoU (mIoU)"
                  << std::setw(25) << (std::to_string(cv_res.mIoU).substr(0, 6))
                  << std::setw(25) << (std::to_string(yolo_res.mIoU).substr(0, 6))
                  << "\n";

        std::cout << std::left << std::setw(30) << " 6-Fold CV Accuracy"
                  << std::setw(25) << (std::to_string(cv_res.cv_accuracy).substr(0, 5) + "%")
                  << std::setw(25) << (std::to_string(yolo_res.cv_accuracy).substr(0, 5) + "%")
                  << "\n";

        std::cout << std::string(80, '-') << "\n";
        std::cout << " Per-Class mIoU Breakdown:\n";
        for (int c = 0; c < 6; ++c) {
            std::cout << "   - " << std::left << std::setw(25) << class_names_arr[c]
                      << std::setw(25) << (std::to_string(cv_res.per_class_miou[c]).substr(0, 6))
                      << std::setw(25) << (std::to_string(yolo_res.per_class_miou[c]).substr(0, 6))
                      << "\n";
        }
        std::cout << "================================================================================\n\n";

    } else {
        // Run single pipeline (YOLO if --use-yolo, else Classical CV)
        runPipeline(raw_dataset, use_yolo, output_dir);
    }

    return 0;
}