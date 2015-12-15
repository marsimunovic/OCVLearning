#include "featureextractor.h"
#include "math_functions.h"
#include "cvline.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include <vector>
#include <iostream>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

FeatureExtractor::FeatureExtractor()
    :n_image_parts(4)
{
}

void FeatureExtractor::import_image(cv::Mat &img_color)
{
    GaussianBlur(img_color, img_color, Size(5, 5), 3, 3);
    src = Mat(src.size(), CV_8UC1);
    cvtColor(img_color, src, COLOR_BGR2GRAY);

}

void FeatureExtractor::extract_image_parameters()
{
    //create placeholder for ROI segments in image
    int src_h = src.size().height;
    int src_w = src.size().width;
    int total_segments = 2*n_image_parts-1;
    int segment_w = src_w/n_image_parts;
    //last segment width my differ if number is not divisible with n_image_parts
    int last_segment_w = src_w - (n_image_parts-1)*segment_w;
    cv::Mat roi_segment[total_segments];

    std::string win_name = "Segment ";
    std::string win_name2 = "Vertical lines ";
    Mat drawing = Mat::zeros(src.size(), CV_8UC3);
    namedWindow(win_name2, 0);
    Scalar colors[3] = {Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255)};
    vector<CVLine> all_vertical;
    for(int i = 0; i < total_segments; ++i)
    {
        //odd segments starts at half of previous even segment
        int segment_start = segment_w/2*i;
        roi_segment[i] = Mat(src, Rect(segment_start, 0,
                         (i < total_segments - 1)? segment_w : last_segment_w, src_h));
        cv::Mat bw_segment(roi_segment[i].size(), CV_8UC1);
        perform_tresholding(roi_segment[i], bw_segment);
        namedWindow(win_name + char(i + 49), 0);
        imshow(win_name + char(i + 49), bw_segment);
        //waitKey();
        //perform analysis of black-n-white segment
        //in order to extract geometrical shapes
        vector<vector<Point> > approx_contours;
        extract_contours(bw_segment, approx_contours);

        //extract verticalish and horizontalish lines from image
        vector<CVLine> vertical;
        vector<CVLine> horizontal;
        extract_horiz_vert_lines(approx_contours, bw_segment.size(), segment_start, vertical, horizontal);

        Scalar color = colors[i%3];
        for(size_t i = 0; i < all_vertical.size(); ++i)
        {
           all_vertical[i].shift_right(segment_start);
           cv::line(drawing, vertical[i].start, vertical[i].end, color, 2);
        }
        imshow(win_name2, drawing);


        waitKey();
        if(!all_vertical.empty())
        {
            all_vertical.insert(all_vertical.end(), vertical.begin(), vertical.end());
        }
        else
        {
            std::swap(all_vertical, vertical);
        }

    }

}

void FeatureExtractor::perform_tresholding(cv::Mat &src, cv::Mat &bw_output)
{
    Mat bw_hist;
    int histSize = 256; //number of bins
    float range[] = {0, 256};
    const float* histRange = {range};
    int channs[] = {0};

    calcHist(&src, 1, channs, Mat(), bw_hist, 1, &histSize, &histRange);
    int min_ind = 130;
    int max_ind = 180;
    float min_val = static_cast<float>(src.size().area());
    //find best position for mode separation
    //TODO: determine mode ranges instead of fixed min_ind and max_ind
    for(int i = min_ind; i < max_ind; ++i)
    {
        if(min_val >= bw_hist.at<float>(i))
        {
            //cout << i << " : " << bw_hist.at<float>(i) << endl;
            min_val = bw_hist.at<float>(i);
            min_ind = i;
        }
    }
    //min_ind is threshold value
    threshold(src, bw_output, min_ind, 255, THRESH_BINARY);

}

void FeatureExtractor::extract_contours(cv::Mat &src, std::vector<std::vector<cv::Point> > &contours)
{
    vector<vector<Point>> real_contours;
    findContours(src, real_contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    size_t min_len = src.size().width/15;

    for(size_t i = 0; i < real_contours.size(); ++i)
    {
        if(real_contours[i].size() < min_len)
            continue;
        vector<Point> brief_contour;
        approxPolyDP(real_contours[i], brief_contour, 20, false);
        contours.push_back(std::move(brief_contour));
    }
}

void FeatureExtractor::extract_horiz_vert_lines(std::vector<std::vector<cv::Point> > &contours,
                                                cv::Size const& src_sz, int offset,
                                 std::vector<CVLine> &vertical, std::vector<CVLine> &horizontal)
{
    int min_x = src_sz.width*0.04;
    int max_x = src_sz.width*0.96;
    int min_vertical_len = src_sz.height*0.07;
    for(auto &contour : contours)
    {
        for(size_t i = 1; i < contour.size(); ++i)
        {
            cv::Point& start = contour[i-1];
            cv::Point& end = contour[i];
            if(start.x < min_x || start.x > max_x || end.x < min_x || end.x > max_x)
                continue;
            CVLine l(start, end);
            l.shift_right(offset);
            if(l.inclination > (M_PI/6))
            {
                if(l.length < min_vertical_len)
                    continue;
                vertical.push_back((std::move(l)));
            }
            else
            {
                horizontal.push_back((std::move(l)));
            }
        }
    }
    cv::Point origin(0, src_sz.height-1);

    std::sort(vertical.begin(), vertical.end(),
              [&origin](CVLine const &first, CVLine const &second)
    {
        return ::line_point_dist(first.start, first.end, origin) <
        ::line_point_dist(second.start, second.end, origin);
    });
    std::sort(horizontal.begin(), horizontal.end(),
              [&origin](CVLine const &first, CVLine const &second)
    {
        return MIN(first.start.x, first.end.x) < MIN(second.start.x, second.end.x);
    });
}
