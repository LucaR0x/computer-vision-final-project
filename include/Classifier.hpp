#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>
#include "FeatureExtractor.hpp"

// Member 2: Action Classifier using SVM and 6-Fold Cross Validation
class ActionClassifier {
private:
    cv::Ptr<cv::ml::SVM> svm_model;
    std::vector<float> trained_means;
    std::vector<float> trained_stds;

    void prepareMatrices(const std::vector<FeatureSample>& samples, cv::Mat& out_features, cv::Mat& out_labels);
    void computeScalingParams(const cv::Mat& data);
    void applyScaling(cv::Mat& data) const;

public:
    ActionClassifier();

    // 6-Fold Stratified Cross Validation Evaluation (returns mean CV accuracy percentage)
    float evaluate(const std::vector<FeatureSample>& dataset, float train_ratio = 0.75f);

    // Train final model and save to xml file
    void trainAndSave(const std::vector<FeatureSample>& dataset, const std::string& model_output_path);

    // Predict action label for a single sequence descriptor vector
    int predict(const std::vector<float>& raw_feature_vector) const;

    // Load pre-trained SVM model
    bool loadModel(const std::string& model_input_path);
};

#endif // CLASSIFIER_HPP