#include "FeatureExtractor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>

FeatureExtractor::FeatureExtractor() {}

// Helper function to calculate percentile of values in OpenCV Mat
static float getMatPercentile(const cv::Mat& mat, const cv::Mat& mask, float pct) {
    std::vector<float> vals;
    vals.reserve(mat.rows * mat.cols);
    for (int r = 0; r < mat.rows; ++r) {
        for (int c = 0; c < mat.cols; ++c) {
            if (mask.empty() || mask.at<uchar>(r, c) > 0) {
                vals.push_back(mat.at<float>(r, c));
            }
        }
    }
    if (vals.empty()) return 0.0f;
    size_t idx = static_cast<size_t>(std::clamp(pct, 0.0f, 1.0f) * (vals.size() - 1));
    std::nth_element(vals.begin(), vals.begin() + idx, vals.end());
    return vals[idx];
}

// Helper function to compute mean of a vector
static float computeVectorMean(const std::vector<float>& vec) {
    if (vec.empty()) return 0.0f;
    float sum = std::accumulate(vec.begin(), vec.end(), 0.0f);
    return sum / static_cast<float>(vec.size());
}

// Helper function to compute max of a vector
static float computeVectorMax(const std::vector<float>& vec) {
    if (vec.empty()) return 0.0f;
    return *std::max_element(vec.begin(), vec.end());
}

// Helper function to compute std dev of a vector
static float computeVectorStd(const std::vector<float>& vec) {
    if (vec.size() < 2) return 0.0f;
    float mean = computeVectorMean(vec);
    float sq_sum = 0.0f;
    for (float val : vec) {
        sq_sum += (val - mean) * (val - mean);
    }
    return std::sqrt(sq_sum / static_cast<float>(vec.size()));
}

// Helper function to count zero-crossings of a signal relative to its mean
static int countZeroCrossings(const std::vector<float>& signal) {
    if (signal.size() < 3) return 0;
    float mean = computeVectorMean(signal);
    int count = 0;
    for (size_t i = 1; i < signal.size(); ++i) {
        float prev = signal[i - 1] - mean;
        float curr = signal[i] - mean;
        if ((prev <= 0.0f && curr > 0.0f) || (prev >= 0.0f && curr < 0.0f)) {
            count++;
        }
    }
    return count;
}

int FeatureExtractor::countSignalPeaks(const std::vector<float>& signal, float threshold) {
    if (signal.size() < 5) return 0;

    std::vector<float> smoothed(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        float sum = signal[i];
        int count = 1;
        if (i > 0) { sum += signal[i - 1]; count++; }
        if (i + 1 < signal.size()) { sum += signal[i + 1]; count++; }
        smoothed[i] = sum / count;
    }

    int peaks = 0;
    for (size_t i = 1; i < smoothed.size() - 1; ++i) {
        if (smoothed[i] > smoothed[i - 1] && smoothed[i] > smoothed[i + 1] && smoothed[i] > threshold) {
            peaks++;
        }
    }
    return peaks;
}

FeatureSample FeatureExtractor::extractFromSequence(const std::vector<cv::Mat>& frames, 
                                                   int label, 
                                                   const std::string& name,
                                                   const cv::Rect& roi) {
    FeatureSample sample;
    sample.label = label;
    sample.sequence_name = name;

    if (frames.size() < 2) return sample;

    int img_w = frames[0].cols;
    int img_h = frames[0].rows;
    int total_frames = static_cast<int>(frames.size());
    int mid_t = total_frames / 2; // Frame 20 (median frame)

    cv::Rect ref_roi = roi;
    if (ref_roi.width <= 10 || ref_roi.height <= 25 || ref_roi.width >= img_w * 0.85 || ref_roi.height >= img_h * 0.85) {
        // Fallback to center region if provided ROI is invalid or corrupted
        ref_roi = cv::Rect(img_w / 4, img_h / 6, img_w / 2, (img_h * 2) / 3);
    }
    ref_roi = ref_roi & cv::Rect(0, 0, img_w, img_h);

    float actor_height = static_cast<float>(ref_roi.height > 15 ? ref_roi.height : img_h * 0.5f);

    // Track bounding box center across frames using sequence tracker
    Tracker seq_tracker;
    seq_tracker.init(frames);

    std::vector<cv::Point2f> centers(total_frames);
    std::vector<cv::Rect> tracked_boxes(total_frames);
    std::vector<bool> actor_present(total_frames, false);

    for (int t = 0; t < total_frames; ++t) {
        cv::Mat mask_t;
        cv::Rect b_t = seq_tracker.processFrame(frames[t], mask_t);

        if (b_t.width > 5 && b_t.height > 5) {
            tracked_boxes[t] = b_t;
            centers[t] = cv::Point2f(b_t.x + b_t.width / 2.0f, b_t.y + b_t.height / 2.0f);
            actor_present[t] = true;
        } else {
            tracked_boxes[t] = ref_roi;
            centers[t] = cv::Point2f(ref_roi.x + ref_roi.width / 2.0f, ref_roi.y + ref_roi.height / 2.0f);
            actor_present[t] = false;
        }
    }

    // Feature vectors over time
    std::vector<float> frame_trans_speeds;   // Centroid horizontal translation speed
    std::vector<float> body_flow_mags;       // Average optical flow magnitude in whole body
    std::vector<float> body_flow_p80s;       // 85th percentile flow speed in body
    std::vector<float> leg_flow_mags;        // Lower body flow magnitude
    std::vector<float> leg_flow_p80s;        // Lower body 85th percentile flow speed
    std::vector<float> upper_flow_mags;      // Upper body flow magnitude
    std::vector<float> upper_vy_mags;        // Upper body vertical velocity magnitude |Vy|
    std::vector<float> upper_vy_means;       // Upper body vertical velocity signed Vy
    std::vector<float> convergence_history;  // Upper body horizontal convergence (Clapping)
    std::vector<float> asymmetry_history;    // Upper body left-right asymmetry (Boxing)
    std::vector<float> punch_peaks;          // Upper body peak horizontal flow (Boxing)
    std::vector<float> aspect_ratios;        // Bounding box aspect ratios
    int opposed_motion_count = 0;
    int upper_motion_frames = 0;

    cv::Mat prev_gray;
    if (frames[0].channels() == 3) {
        cv::cvtColor(frames[0], prev_gray, cv::COLOR_BGR2GRAY);
    } else {
        prev_gray = frames[0];
    }

    int active_count = 0;
    for (int t = 0; t < total_frames; ++t) {
        if (actor_present[t]) active_count++;
        if (t == 0) continue;

        cv::Mat curr_gray;
        if (frames[t].channels() == 3) {
            cv::cvtColor(frames[t], curr_gray, cv::COLOR_BGR2GRAY);
        } else {
            curr_gray = frames[t];
        }

        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prev_gray, curr_gray, flow, 0.5, 4, 15, 3, 5, 1.2, 0);

        cv::Mat flow_channels[2];
        cv::split(flow, flow_channels); // 0: Vx, 1: Vy

        cv::Mat mag;
        cv::magnitude(flow_channels[0], flow_channels[1], mag);

        // Analyze motion inside current tracked actor ROI
        if (actor_present[t]) {
            cv::Rect box = tracked_boxes[t] & cv::Rect(0, 0, img_w, img_h);
            if (box.width > 5 && box.height > 5) {
                cv::Mat box_mag = mag(box);
                cv::Mat box_vx  = flow_channels[0](box);
                cv::Mat box_vy  = flow_channels[1](box);

                // 1. Centroid translation velocity (px/frame normalized by actor height)
                if (actor_present[t - 1]) {
                    float dx = centers[t].x - centers[t - 1].x;
                    frame_trans_speeds.push_back(std::abs(dx) / actor_height);
                }

                // 2. Whole body flow speeds
                float mean_mag = static_cast<float>(cv::mean(box_mag)[0]);
                body_flow_mags.push_back(mean_mag / actor_height);

                cv::Mat abs_box_vx = cv::abs(box_vx);
                float p80_vx = getMatPercentile(abs_box_vx, cv::Mat(), 0.85f);
                body_flow_p80s.push_back(p80_vx / actor_height);

                // Sub-divide ROI into Upper body and Lower body
                int bh = box.height;
                int bw = box.width;
                int half_h = bh / 2;
                int half_w = bw / 2;

                cv::Rect top_rect(0, 0, bw, half_h);
                cv::Mat top_mag = box_mag(top_rect);
                cv::Mat top_vx  = box_vx(top_rect);
                cv::Mat top_vy  = box_vy(top_rect);

                cv::Rect bot_rect(0, half_h, bw, bh - half_h);
                cv::Mat bot_mag = box_mag(bot_rect);
                cv::Mat bot_vx  = box_vx(bot_rect);

                float top_mag_sum = static_cast<float>(cv::sum(top_mag)[0]);
                float bot_mag_sum = static_cast<float>(cv::sum(bot_mag)[0]);
                upper_flow_mags.push_back(top_mag_sum);
                leg_flow_mags.push_back(bot_mag_sum);

                cv::Mat abs_bot_vx = cv::abs(bot_vx);
                float leg_p80 = getMatPercentile(abs_bot_vx, cv::Mat(), 0.85f);
                leg_flow_p80s.push_back(leg_p80 / actor_height);

                cv::Mat abs_top_vx = cv::abs(top_vx);
                float punch_p90 = getMatPercentile(abs_top_vx, cv::Mat(), 0.90f);
                punch_peaks.push_back(punch_p90 / actor_height);

                cv::Mat abs_top_vy = cv::abs(top_vy);
                float mean_top_vy_abs = static_cast<float>(cv::mean(abs_top_vy)[0]);
                upper_vy_mags.push_back(mean_top_vy_abs / actor_height);

                float mean_top_vy = static_cast<float>(cv::mean(top_vy)[0]);
                upper_vy_means.push_back(mean_top_vy);

                // Upper body convergence (Clapping) and asymmetry (Boxing)
                if (half_w > 2) {
                    cv::Mat top_left_vx = top_vx(cv::Rect(0, 0, half_w, half_h));
                    cv::Mat top_right_vx = top_vx(cv::Rect(half_w, 0, bw - half_w, half_h));

                    float left_m = static_cast<float>(cv::mean(top_left_vx)[0]);
                    float right_m = static_cast<float>(cv::mean(top_right_vx)[0]);
                    // Left moving right (>0) and right moving left (<0) means convergence (>0)
                    convergence_history.push_back((left_m - right_m) / actor_height);

                    if (std::abs(left_m) > 0.05f || std::abs(right_m) > 0.05f) {
                        upper_motion_frames++;
                        if (left_m * right_m < 0.0f) {
                            opposed_motion_count++;
                        }
                    }

                    float left_e = static_cast<float>(cv::sum(cv::abs(top_left_vx))[0]);
                    float right_e = static_cast<float>(cv::sum(cv::abs(top_right_vx))[0]);
                    float asym = std::abs(left_e - right_e) / (left_e + right_e + 1e-4f);
                    asymmetry_history.push_back(asym);
                }

                aspect_ratios.push_back(static_cast<float>(bw) / static_cast<float>(bh));
            }
        }

        prev_gray = curr_gray;
    }

    // --- Compute aggregated feature metrics ---
    float active_ratio = static_cast<float>(active_count) / static_cast<float>(total_frames);

    // 1-4: Net sequence displacement & translation speed metrics (Key for Walking vs Jogging vs Running!)
    int first_active = -1, last_active = -1;
    for (int t = 0; t < total_frames; ++t) {
        if (actor_present[t]) {
            if (first_active == -1) first_active = t;
            last_active = t;
        }
    }
    float total_displacement = 0.0f;
    if (first_active != -1 && last_active > first_active) {
        float dx_net = std::abs(centers[last_active].x - centers[first_active].x);
        int duration = (last_active - first_active);
        total_displacement = (dx_net / (actor_height + 1e-4f)) / static_cast<float>(duration);
    }

    float mean_trans_speed = computeVectorMean(frame_trans_speeds);
    float max_trans_speed  = computeVectorMax(frame_trans_speeds);
    float mean_body_p80    = computeVectorMean(body_flow_p80s);
    float max_body_p80     = computeVectorMax(body_flow_p80s);

    // 5-7: Leg motion metrics
    float mean_leg_p80  = computeVectorMean(leg_flow_p80s);
    float max_leg_p80   = computeVectorMax(leg_flow_p80s);
    float leg_energy_std = computeVectorStd(leg_flow_mags);
    int stride_crossings = countZeroCrossings(leg_flow_mags);

    // 8-11: Upper body vertical & energy metrics (Key for Waving!)
    float sum_top_energy = std::accumulate(upper_flow_mags.begin(), upper_flow_mags.end(), 0.0f);
    float sum_bot_energy = std::accumulate(leg_flow_mags.begin(), leg_flow_mags.end(), 0.0f);
    float vertical_bias  = sum_top_energy / (sum_bot_energy + 1e-4f);

    float mean_upper_vy_mag  = computeVectorMean(upper_vy_mags);
    float max_upper_vy_mag   = computeVectorMax(upper_vy_mags);
    int vy_zero_crossings    = countZeroCrossings(upper_vy_means);

    // 12-16: Convergence, Opposed Motion & Boxing metrics
    float conv_std = computeVectorStd(convergence_history);
    float conv_max = computeVectorMax(convergence_history);
    float opposed_motion_ratio = upper_motion_frames > 0 ? (static_cast<float>(opposed_motion_count) / upper_motion_frames) : 0.0f;
    float punch_peak_max = computeVectorMax(punch_peaks);
    float avg_asymmetry = computeVectorMean(asymmetry_history);

    // 17: Aspect ratio
    float max_aspect_ratio = computeVectorMax(aspect_ratios);

    // Build 18-element feature vector
    sample.descriptors.push_back(total_displacement);                       // 1. Total net sequence horizontal displacement
    sample.descriptors.push_back(max_trans_speed);                          // 2. Max translation speed
    sample.descriptors.push_back(mean_body_p80);                            // 3. Mean body 85th percentile flow speed
    sample.descriptors.push_back(max_body_p80);                             // 4. Max body 85th percentile flow speed
    sample.descriptors.push_back(mean_leg_p80);                             // 5. Mean leg flow speed
    sample.descriptors.push_back(max_leg_p80);                              // 6. Max leg flow speed
    sample.descriptors.push_back(leg_energy_std);                           // 7. Leg motion energy std dev (stride rhythm)
    sample.descriptors.push_back(static_cast<float>(stride_crossings));     // 8. Stride zero crossings (gait frequency)
    sample.descriptors.push_back(vertical_bias);                            // 9. Upper vs lower body energy ratio
    sample.descriptors.push_back(mean_upper_vy_mag);                        // 10. Upper body vertical motion magnitude
    sample.descriptors.push_back(max_upper_vy_mag);                         // 11. Upper body max vertical motion magnitude
    sample.descriptors.push_back(static_cast<float>(vy_zero_crossings));    // 12. Upper body vertical oscillation count (Waving)
    sample.descriptors.push_back(conv_std);                                 // 13. Convergence std dev (Clapping)
    sample.descriptors.push_back(conv_max);                                 // 14. Peak convergence speed (Clapping)
    sample.descriptors.push_back(opposed_motion_ratio);                    // 15. Opposed left-right motion ratio (Clapping)
    sample.descriptors.push_back(punch_peak_max);                           // 16. Peak horizontal punch speed (Boxing)
    sample.descriptors.push_back(avg_asymmetry);                            // 17. Upper body motion asymmetry (Boxing)
    sample.descriptors.push_back(max_aspect_ratio);                         // 18. Bounding box max aspect ratio

    return sample;
}