#include "Classifier.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cmath>

ActionClassifier::ActionClassifier() {
    svm_model = cv::ml::SVM::create();
    svm_model->setType(cv::ml::SVM::C_SVC);
    svm_model->setKernel(cv::ml::SVM::RBF);
    svm_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 5000, 1e-6));
}

void ActionClassifier::prepareMatrices(const std::vector<FeatureSample>& samples, cv::Mat& out_features, cv::Mat& out_labels) {
    int num_samples = static_cast<int>(samples.size());
    int num_features = static_cast<int>(samples[0].descriptors.size());

    out_features = cv::Mat(num_samples, num_features, CV_32F);
    out_labels = cv::Mat(num_samples, 1, CV_32S);

    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            out_features.at<float>(i, j) = samples[i].descriptors[j];
        }
        out_labels.at<int>(i, 0) = samples[i].label;
    }
}

void ActionClassifier::computeScalingParams(const cv::Mat& data) {
    int cols = data.cols;
    int rows = data.rows;
    trained_means.assign(cols, 0.0f);
    trained_stds.assign(cols, 0.0f);

    for (int j = 0; j < cols; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < rows; ++i) {
            sum += data.at<float>(i, j);
        }
        trained_means[j] = sum / rows;

        float sq_diff = 0.0f;
        for (int i = 0; i < rows; ++i) {
            sq_diff += std::pow(data.at<float>(i, j) - trained_means[j], 2.0f);
        }
        trained_stds[j] = std::sqrt(sq_diff / rows) + 1e-6f;
    }
}

void ActionClassifier::applyScaling(cv::Mat& data) const {
    for (int i = 0; i < data.rows; ++i) {
        for (int j = 0; j < data.cols; ++j) {
            data.at<float>(i, j) = (data.at<float>(i, j) - trained_means[j]) / trained_stds[j];
        }
    }
}

void ActionClassifier::evaluate(const std::vector<FeatureSample>& dataset, float) {
    if (dataset.empty()) return;

    std::vector<FeatureSample> class_buckets[6];
    for (const auto& s : dataset) {
        if (s.label >= 1 && s.label <= 6) {
            class_buckets[s.label - 1].push_back(s);
        }
    }

    std::mt19937 g(42);
    for (int c = 0; c < 6; ++c) {
        std::shuffle(class_buckets[c].begin(), class_buckets[c].end(), g);
    }

    const int K_FOLDS = 4;
    int global_confusion_matrix[6][6] = {0};
    int total_evaluated = 0;
    int total_correct = 0;

    cv::ml::ParamGrid c_grid(0.1, 100, 10);
    cv::ml::ParamGrid gamma_grid(0.001, 1.0, 5);

    const std::string class_names[6] = {"Boxing", "Clapping", "Waving", "Jogging", "Running", "Walking"};

    for (int fold = 0; fold < K_FOLDS; ++fold) {
        std::vector<FeatureSample> train_fold, test_fold;

        for (int c = 0; c < 6; ++c) {
            int fold_size = static_cast<int>(class_buckets[c].size()) / K_FOLDS;
            int start_idx = fold * fold_size;
            int end_idx = start_idx + fold_size;

            for (int i = 0; i < static_cast<int>(class_buckets[c].size()); ++i) {
                if (i >= start_idx && i < end_idx) test_fold.push_back(class_buckets[c][i]);
                else train_fold.push_back(class_buckets[c][i]);
            }
        }

        cv::Mat train_X, train_y, test_X, test_y;
        prepareMatrices(train_fold, train_X, train_y);
        prepareMatrices(test_fold, test_X, test_y);

        computeScalingParams(train_X);
        applyScaling(train_X);
        applyScaling(test_X);

        cv::Ptr<cv::ml::SVM> fold_svm = cv::ml::SVM::create();
        fold_svm->setType(cv::ml::SVM::C_SVC);
        fold_svm->setKernel(cv::ml::SVM::RBF);

        cv::ml::ParamGrid c_grid(0.1, 100, 10);
        cv::ml::ParamGrid gamma_grid(0.001, 1.0, 5);

        fold_svm->trainAuto(cv::ml::TrainData::create(train_X, cv::ml::ROW_SAMPLE, train_y),
                             10, c_grid, gamma_grid,
                             cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P),
                             cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU),
                             cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF),
                             cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE),
                             true);

        for (int i = 0; i < test_X.rows; ++i) {
            int pred = static_cast<int>(fold_svm->predict(test_X.row(i)));
            int gt = test_y.at<int>(i, 0);

            if (gt >= 1 && gt <= 6 && pred >= 1 && pred <= 6) {
                global_confusion_matrix[gt - 1][pred - 1]++;
                if (pred == gt) {
                    total_correct++;
                } else {
                    std::cout << "[MISCLASS] Fold " << fold << " | Seq: " 
                              << test_fold[i].sequence_name << " (GT: " << gt 
                              << " " << class_names[gt - 1] << " -> Pred: " << pred 
                              << " " << class_names[pred - 1] << ")\n";
                }
                total_evaluated++;
            }
        }
    }

    float accuracy = (static_cast<float>(total_correct) / total_evaluated) * 100.0f;
    std::cout << "\n================ 4-FOLD CV EVALUATION METRICS ================\n";
    std::cout << "Total Evaluated Sequences: " << total_evaluated << " / 72\n";
    std::cout << "Mean Cross-Validation Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%\n\n";

    std::cout << "Cumulative Confusion Matrix:\n";
    std::cout << "      [1]  [2]  [3]  [4]  [5]  [6]\n";
    for (int r = 0; r < 6; ++r) {
        std::cout << "[" << (r + 1) << "]  ";
        for (int c = 0; c < 6; ++c) {
            std::cout << std::setw(4) << global_confusion_matrix[r][c] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nPer-Class Cumulative Metrics:\n";
    for (int c = 0; c < 6; ++c) {
        int tp = global_confusion_matrix[c][c];
        int fn = 0, fp = 0;

        for (int i = 0; i < 6; ++i) {
            if (i != c) {
                fn += global_confusion_matrix[c][i];
                fp += global_confusion_matrix[i][c];
            }
        }

        float precision = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
        float recall = (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;
        float f1 = (precision + recall > 0) ? 2.0f * (precision * recall) / (precision + recall) : 0.0f;

        std::cout << "Class " << (c + 1) << " (" << std::setw(9) << class_names[c] << ") -> "
                  << "Prec: " << std::setprecision(2) << precision << " | "
                  << "Rec: " << recall << " | "
                  << "F1: " << f1 << "\n";
    }
    std::cout << "==============================================================\n\n";
}

void ActionClassifier::trainAndSave(const std::vector<FeatureSample>& dataset, const std::string& model_output_path) {
    cv::Mat all_X, all_y;
    prepareMatrices(dataset, all_X, all_y);

    computeScalingParams(all_X);
    applyScaling(all_X);

    cv::ml::ParamGrid c_grid(0.1, 100, 10);
    cv::ml::ParamGrid gamma_grid(0.001, 1.0, 5);

    svm_model->trainAuto(cv::ml::TrainData::create(all_X, cv::ml::ROW_SAMPLE, all_y),
                         10, c_grid, gamma_grid,
                         cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P),
                         cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU),
                         cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF),
                         cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE),
                         true);

    svm_model->save(model_output_path);

    std::string scale_path = model_output_path + ".scale.yaml";
    cv::FileStorage fs(scale_path, cv::FileStorage::WRITE);
    fs << "means" << trained_means;
    fs << "stds" << trained_stds;
    fs.release();

    std::cout << "[INFO] Final SVM Model and Scaling Params saved to: " << model_output_path << std::endl;
}

int ActionClassifier::predict(const std::vector<float>& raw_feature_vector) const {
    if (trained_means.empty() || trained_stds.empty()) {
        std::cerr << "[ERROR] Scaler params not initialized before predict!" << std::endl;
        return -1;
    }

    cv::Mat sample_mat(1, static_cast<int>(raw_feature_vector.size()), CV_32F);
    for (size_t j = 0; j < raw_feature_vector.size(); ++j) {
        sample_mat.at<float>(0, static_cast<int>(j)) = (raw_feature_vector[j] - trained_means[j]) / trained_stds[j];
    }

    return static_cast<int>(svm_model->predict(sample_mat));
}

bool ActionClassifier::loadModel(const std::string& model_input_path) {
    svm_model = cv::ml::SVM::load(model_input_path);

    std::string scale_path = model_input_path + ".scale.yaml";
    cv::FileStorage fs(scale_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[WARNING] Could not find scaling file: " << scale_path << std::endl;
        return !svm_model.empty();
    }

    fs["means"] >> trained_means;
    fs["stds"] >> trained_stds;
    fs.release();

    return !svm_model.empty();
}