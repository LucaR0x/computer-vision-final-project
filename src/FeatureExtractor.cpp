#include "FeatureExtractor.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>

FeatureExtractor::FeatureExtractor() {}

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
    int total_frames = (int)frames.size();

    cv::Rect ref_roi = roi;
    if (ref_roi.width <= 10 || ref_roi.height <= 25 || ref_roi.width >= img_w * 0.85 || ref_roi.height >= img_h * 0.85) {
        ref_roi = cv::Rect(img_w / 4, img_h / 6, img_w / 2, (img_h * 2) / 3);
    }
    ref_roi = ref_roi & cv::Rect(0, 0, img_w, img_h);

    float actor_height = (float)(ref_roi.height > 15 ? ref_roi.height : img_h * 0.5f);

    // Sequence tracker to follow actor center across frames
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

    // 3-frame moving average filter on centers to suppress single-frame arm/box expansion jitter
    std::vector<cv::Point2f> smooth_centers = centers;
    for (int t = 1; t < total_frames - 1; ++t) {
        if (actor_present[t - 1] && actor_present[t] && actor_present[t + 1]) {
            smooth_centers[t].x = (centers[t - 1].x + centers[t].x + centers[t + 1].x) / 3.0f;
            smooth_centers[t].y = (centers[t - 1].y + centers[t].y + centers[t + 1].y) / 3.0f;
        }
    }

    std::vector<float> frame_trans_speeds;
    std::vector<float> body_flow_mags;
    std::vector<float> body_flow_p80s;
    std::vector<float> leg_flow_mags;
    std::vector<float> leg_flow_p80s;
    std::vector<float> upper_flow_mags;
    std::vector<float> upper_vy_mags;
    std::vector<float> upper_vy_means;
    std::vector<float> convergence_history;
    std::vector<float> asymmetry_history;
    std::vector<float> punch_peaks;
    std::vector<float> aspect_ratios;
    std::vector<float> head_energy_ratios;
    std::vector<float> box_widths;

    int opposed_motion_count = 0;
    int upper_motion_frames = 0;

    cv::Mat prev_gray;
    if (frames[0].channels() == 3) {
        cv::cvtColor(frames[0], prev_gray, cv::COLOR_BGR2GRAY);
    } else {
        prev_gray = frames[0].clone();
    }

    for (int t = 0; t < total_frames; ++t) {
        if (t == 0) continue;

        cv::Mat curr_gray;
        if (frames[t].channels() == 3) {
            cv::cvtColor(frames[t], curr_gray, cv::COLOR_BGR2GRAY);
        } else {
            curr_gray = frames[t].clone();
        }

        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prev_gray, curr_gray, flow, 0.5, 4, 15, 3, 5, 1.2, 0);

        cv::Mat flow_channels[2];
        cv::split(flow, flow_channels); // 0: Vx, 1: Vy

        cv::Mat mag;
        cv::magnitude(flow_channels[0], flow_channels[1], mag);

        if (actor_present[t]) {
            cv::Rect box = tracked_boxes[t] & cv::Rect(0, 0, img_w, img_h);
            if (box.width > 5 && box.height > 5) {
                cv::Mat box_mag = mag(box);
                cv::Mat box_vx  = flow_channels[0](box);
                cv::Mat box_vy  = flow_channels[1](box);

                // 1. Horizontal centroid translation speed (using 3-frame smoothed centers)
                if (actor_present[t - 1]) {
                    float dx = smooth_centers[t].x - smooth_centers[t - 1].x;
                    frame_trans_speeds.push_back(std::abs(dx) / actor_height);
                }

                // 2. Whole body flow speeds
                float mean_mag = (float)cv::mean(box_mag)[0];
                body_flow_mags.push_back(mean_mag / actor_height);

                cv::Mat abs_box_vx = cv::abs(box_vx);
                std::vector<float> vx_vals;
                for (int r = 0; r < abs_box_vx.rows; ++r) {
                    for (int c = 0; c < abs_box_vx.cols; ++c) {
                        vx_vals.push_back(abs_box_vx.at<float>(r, c));
                    }
                }
                if (!vx_vals.empty()) {
                    int p85_idx = (int)(0.85f * (vx_vals.size() - 1));
                    std::nth_element(vx_vals.begin(), vx_vals.begin() + p85_idx, vx_vals.end());
                    body_flow_p80s.push_back(vx_vals[p85_idx] / actor_height);
                }

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

                float top_mag_sum = (float)cv::sum(top_mag)[0];
                float bot_mag_sum = (float)cv::sum(bot_mag)[0];
                upper_flow_mags.push_back(top_mag_sum);
                leg_flow_mags.push_back(bot_mag_sum);

                cv::Mat abs_bot_vx = cv::abs(bot_vx);
                std::vector<float> bot_vx_vals;
                for (int r = 0; r < abs_bot_vx.rows; ++r) {
                    for (int c = 0; c < abs_bot_vx.cols; ++c) {
                        bot_vx_vals.push_back(abs_bot_vx.at<float>(r, c));
                    }
                }
                if (!bot_vx_vals.empty()) {
                    int p85_idx = (int)(0.85f * (bot_vx_vals.size() - 1));
                    std::nth_element(bot_vx_vals.begin(), bot_vx_vals.begin() + p85_idx, bot_vx_vals.end());
                    leg_flow_p80s.push_back(bot_vx_vals[p85_idx] / actor_height);
                }

                cv::Mat abs_top_vx = cv::abs(top_vx);
                std::vector<float> top_vx_vals;
                for (int r = 0; r < abs_top_vx.rows; ++r) {
                    for (int c = 0; c < abs_top_vx.cols; ++c) {
                        top_vx_vals.push_back(abs_top_vx.at<float>(r, c));
                    }
                }
                if (!top_vx_vals.empty()) {
                    int p90_idx = (int)(0.90f * (top_vx_vals.size() - 1));
                    std::nth_element(top_vx_vals.begin(), top_vx_vals.begin() + p90_idx, top_vx_vals.end());
                    punch_peaks.push_back(top_vx_vals[p90_idx] / actor_height);
                }

                cv::Mat abs_top_vy = cv::abs(top_vy);
                float mean_top_vy_abs = (float)cv::mean(abs_top_vy)[0];
                upper_vy_mags.push_back(mean_top_vy_abs / actor_height);

                float mean_top_vy = (float)cv::mean(top_vy)[0];
                upper_vy_means.push_back(mean_top_vy);

                // Top 30% head/hand elevation rect
                int head_h = std::max(1, (int)(bh * 0.3f));
                cv::Rect head_rect(0, 0, bw, head_h);
                cv::Mat head_mag = box_mag(head_rect);
                float head_mag_sum = (float)cv::sum(head_mag)[0];
                float total_box_mag_sum = top_mag_sum + bot_mag_sum + 1e-4f;
                head_energy_ratios.push_back(head_mag_sum / total_box_mag_sum);

                // Upper body convergence (Clapping) and asymmetry (Boxing)
                if (half_w > 2) {
                    cv::Mat top_left_vx = top_vx(cv::Rect(0, 0, half_w, half_h));
                    cv::Mat top_right_vx = top_vx(cv::Rect(half_w, 0, bw - half_w, half_h));

                    float left_m = (float)cv::mean(top_left_vx)[0];
                    float right_m = (float)cv::mean(top_right_vx)[0];
                    convergence_history.push_back((left_m - right_m) / actor_height);

                    if (std::abs(left_m) > 0.05f || std::abs(right_m) > 0.05f) {
                        upper_motion_frames++;
                        if (left_m * right_m < 0.0f) {
                            opposed_motion_count++;
                        }
                    }

                    float left_e = (float)cv::sum(cv::abs(top_left_vx))[0];
                    float right_e = (float)cv::sum(cv::abs(top_right_vx))[0];
                    float asym = std::abs(left_e - right_e) / (left_e + right_e + 1e-4f);
                    asymmetry_history.push_back(asym);
                }

                aspect_ratios.push_back((float)bw / (float)bh);
                box_widths.push_back((float)bw / actor_height);
            }
        }

        prev_gray = curr_gray;
    }

    // --- Compute aggregated feature vector ---
    int first_active = -1, last_active = -1;
    std::vector<float> active_center_ys;
    float min_cx = 99999.0f, max_cx = -99999.0f;
    float cumulative_dx = 0.0f;
    int active_count = 0;

    for (int t = 0; t < total_frames; ++t) {
        if (actor_present[t]) {
            if (first_active == -1) first_active = t;
            last_active = t;
            active_center_ys.push_back(smooth_centers[t].y / actor_height);

            if (smooth_centers[t].x < min_cx) min_cx = smooth_centers[t].x;
            if (smooth_centers[t].x > max_cx) max_cx = smooth_centers[t].x;
            active_count++;

            if (t > 0 && actor_present[t - 1]) {
                cumulative_dx += std::abs(smooth_centers[t].x - smooth_centers[t - 1].x);
            }
        }
    }

    float total_displacement = 0.0f;
    if (first_active != -1 && last_active > first_active) {
        float dx_net = std::abs(centers[last_active].x - centers[first_active].x);
        int duration = (last_active - first_active);
        total_displacement = (dx_net / (actor_height + 1e-4f)) / (float)duration;
    }

    float x_span_norm = 0.0f;
    float cumulative_dx_norm = 0.0f;
    if (active_count > 0 && min_cx < max_cx) {
        x_span_norm = (max_cx - min_cx) / actor_height;
        cumulative_dx_norm = (cumulative_dx / actor_height) / (float)active_count;
    }

    // Direct loop helper for mean, std, max
    auto vec_mean = [](const std::vector<float>& v) -> float {
        if (v.empty()) return 0.0f;
        float s = 0.0f;
        for (float x : v) s += x;
        return s / v.size();
    };

    auto vec_max = [](const std::vector<float>& v) -> float {
        if (v.empty()) return 0.0f;
        float m = v[0];
        for (float x : v) if (x > m) m = x;
        return m;
    };

    auto vec_std = [&](const std::vector<float>& v) -> float {
        if (v.size() < 2) return 0.0f;
        float m = vec_mean(v);
        float sq = 0.0f;
        for (float x : v) sq += (x - m) * (x - m);
        return std::sqrt(sq / v.size());
    };

    auto vec_zero_crossings = [&](const std::vector<float>& v) -> int {
        if (v.size() < 3) return 0;
        float m = vec_mean(v);
        int count = 0;
        for (size_t i = 1; i < v.size(); ++i) {
            float p = v[i - 1] - m;
            float c = v[i] - m;
            if ((p <= 0.0f && c > 0.0f) || (p >= 0.0f && c < 0.0f)) count++;
        }
        return count;
    };

    // Helper to compute percentile of vector
    auto vec_percentile = [](std::vector<float> v, float pct) -> float {
        if (v.empty()) return 0.0f;
        int idx = (int)(std::clamp(pct, 0.0f, 1.0f) * (v.size() - 1));
        std::nth_element(v.begin(), v.begin() + idx, v.end());
        return v[idx];
    };

    float centroid_vertical_bounce = vec_std(active_center_ys);

    float mean_trans_speed = vec_mean(frame_trans_speeds);
    float max_trans_speed  = vec_max(frame_trans_speeds);
    float p75_trans_speed  = vec_percentile(frame_trans_speeds, 0.75f);
    float p90_trans_speed  = vec_percentile(frame_trans_speeds, 0.90f);

    float mean_body_p80    = vec_mean(body_flow_p80s);
    float max_body_p80     = vec_max(body_flow_p80s);

    float mean_leg_p80   = vec_mean(leg_flow_p80s);
    float max_leg_p80    = vec_max(leg_flow_p80s);
    float leg_energy_std = vec_std(leg_flow_mags);
    float leg_mean_energy = vec_mean(leg_flow_mags);
    float leg_energy_cv  = leg_energy_std / (leg_mean_energy + 1e-4f);
    int stride_crossings = vec_zero_crossings(leg_flow_mags);

    float sum_top_energy = 0.0f, sum_bot_energy = 0.0f;
    for (float x : upper_flow_mags) sum_top_energy += x;
    for (float x : leg_flow_mags)   sum_bot_energy += x;
    float vertical_bias  = sum_top_energy / (sum_bot_energy + 1e-4f);

    float mean_upper_vy_mag  = vec_mean(upper_vy_mags);
    float max_upper_vy_mag   = vec_max(upper_vy_mags);
    int vy_zero_crossings    = vec_zero_crossings(upper_vy_means);
    float mean_head_energy_ratio = vec_mean(head_energy_ratios);

    float conv_std = vec_std(convergence_history);
    float conv_max = vec_max(convergence_history);
    float opposed_motion_ratio = upper_motion_frames > 0 ? ((float)opposed_motion_count / upper_motion_frames) : 0.0f;
    float punch_peak_max = vec_max(punch_peaks);
    float avg_asymmetry = vec_mean(asymmetry_history);
    float box_width_std = vec_std(box_widths);

    float max_aspect_ratio = vec_max(aspect_ratios);

    // Build 27-element feature vector
    sample.descriptors.push_back(total_displacement);                       // 1. Total net sequence horizontal displacement
    sample.descriptors.push_back(x_span_norm);                              // 2. Maximum horizontal span of bounding box centroid (Static vs Dynamic)
    sample.descriptors.push_back(cumulative_dx_norm);                       // 3. Cumulative frame-to-frame path distance (Static vs Dynamic)
    sample.descriptors.push_back(max_trans_speed);                          // 4. Max translation speed
    sample.descriptors.push_back(p75_trans_speed);                          // 5. 75th percentile instantaneous speed (Walking vs Jogging vs Running)
    sample.descriptors.push_back(p90_trans_speed);                          // 6. 90th percentile instantaneous speed (Walking vs Jogging vs Running)
    sample.descriptors.push_back(mean_trans_speed);                         // 7. Mean instantaneous speed
    sample.descriptors.push_back(mean_body_p80);                            // 8. Mean body 85th percentile flow speed
    sample.descriptors.push_back(max_body_p80);                             // 9. Max body 85th percentile flow speed
    sample.descriptors.push_back(mean_leg_p80);                             // 10. Mean leg flow speed
    sample.descriptors.push_back(max_leg_p80);                              // 11. Max leg flow speed
    sample.descriptors.push_back(leg_energy_std);                           // 12. Leg motion energy std dev (stride rhythm)
    sample.descriptors.push_back(leg_energy_cv);                            // 13. Leg motion energy coefficient of variation
    sample.descriptors.push_back((float)stride_crossings);     // 14. Stride zero crossings (gait frequency)
    sample.descriptors.push_back(centroid_vertical_bounce);                 // 15. Centroid vertical bounce (Running vs Walking)
    sample.descriptors.push_back(vertical_bias);                            // 16. Upper vs lower body energy ratio
    sample.descriptors.push_back(mean_upper_vy_mag);                        // 17. Upper body vertical motion magnitude
    sample.descriptors.push_back(max_upper_vy_mag);                         // 18. Upper body max vertical motion magnitude
    sample.descriptors.push_back(mean_head_energy_ratio);                   // 19. Top 30% head/hand elevation energy ratio (Waving)
    sample.descriptors.push_back((float)vy_zero_crossings);    // 20. Upper body vertical oscillation count (Waving)
    sample.descriptors.push_back(conv_std);                                 // 21. Convergence std dev (Clapping)
    sample.descriptors.push_back(conv_max);                                 // 22. Peak convergence speed (Clapping)
    sample.descriptors.push_back(opposed_motion_ratio);                    // 23. Opposed left-right motion ratio (Clapping)
    sample.descriptors.push_back(punch_peak_max);                           // 24. Peak horizontal punch speed (Boxing)
    sample.descriptors.push_back(avg_asymmetry);                            // 25. Upper body motion asymmetry (Boxing)
    sample.descriptors.push_back(box_width_std);                            // 26. Bounding box width oscillation std dev (Clapping)
    sample.descriptors.push_back(max_aspect_ratio);                         // 27. Bounding box max aspect ratio

    return sample;
}