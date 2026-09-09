#pragma once

#include "FeatureExtractor.hpp"
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>

class ActionClassifier {
public:
    ActionClassifier();

    void evaluate(const std::vector<FeatureSample>& dataset, float train_ratio = 0.75f);
    void trainAndSave(const std::vector<FeatureSample>& dataset, const std::string& model_output_path);
    int predict(const std::vector<float>& raw_feature_vector) const;
    bool loadModel(const std::string& model_input_path);

private:
    cv::Ptr<cv::ml::SVM> svm_model;
    std::vector<float> trained_means;
    std::vector<float> trained_stds;

    void prepareMatrices(const std::vector<FeatureSample>& samples, cv::Mat& out_features, cv::Mat& out_labels);
    void computeScalingParams(const cv::Mat& data);
    void applyScaling(cv::Mat& data) const;
};