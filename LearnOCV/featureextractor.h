#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include "cvline.h"

#include <opencv2/core/core.hpp>

class FeatureExtractor
{
public:
    FeatureExtractor();
    void import_image(cv::Mat& img_color);
    void extract_image_parameters();
private:

    std::vector<float> detect_key_candidates(std::vector<CVLine>& vertical_lines, cv::Size sz);
    std::vector<float> find_best_edges(std::vector<CVLine>& vertical_lines);
    void perform_tresholding(cv::Mat& src, cv::Mat &bw_output);
    void extract_contours(cv::Mat& src, std::vector<std::vector<cv::Point>> &contours);
    void extract_horiz_vert_lines(std::vector<std::vector<cv::Point>> &contours,
                                  cv::Size const& src_sz, int offset,
                       std::vector<CVLine>& vertical, std::vector<CVLine>& horizontal);
    void vertical_feature_extraction();

private:
    cv::Mat src;
    float global_inclination;
    int   n_line_groups;
    int   n_image_parts; //must be even number
};

#endif // FEATUREEXTRACTOR_H
