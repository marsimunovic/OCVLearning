#include "featureextractor.h"
#include "math_functions.h"
#include "cvline.h"
#include "keycandidate.h"
#include "key.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include <map>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <queue>

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
    Scalar colors[4] = {Scalar(255, 0, 0), Scalar(0, 255, 0),
                        Scalar(0, 0, 255), Scalar(255, 255, 255)};
    vector<CVLine> all_vertical;
    cv::Point origin(-src_w + 1, src_h);
    for(int i = 0; i < total_segments; ++i)
    {
        //odd segments starts at half of previous even segment
        int segment_start = segment_w/2*i;
        roi_segment[i] = Mat(src, Rect(segment_start, 0,
                         (i < total_segments - 1)? segment_w : last_segment_w, src_h));
        cv::Mat bw_segment(roi_segment[i].size(), CV_8UC1);
        perform_tresholding(roi_segment[i], bw_segment);
        //namedWindow(win_name + char(i + 49), 0);
        //imshow(win_name + char(i + 49), bw_segment);
        //waitKey();
        //perform analysis of black-n-white segment
        //in order to extract geometrical shapes
        vector<vector<Point> > approx_contours;
        extract_contours(bw_segment, approx_contours);

        //extract verticalish and horizontalish lines from image
        vector<CVLine> vertical;
        vector<CVLine> horizontal;
        extract_horiz_vert_lines(approx_contours, bw_segment.size(), segment_start, vertical, horizontal);
//        for(size_t i = 0; i < vertical.size(); ++i)
//        {
//            Scalar color = colors[i%3];
//            cv::line(drawing, vertical[i].start, vertical[i].end, color, 2);
//           // cout << all_vertical[i].length << " " << endl;
//            cout << ::line_point_dist(vertical[i].start, vertical[i].end, origin) << endl;
//            imshow(win_name2, drawing);
//            waitKey();
//        }

        if(!all_vertical.empty())
            all_vertical.insert(all_vertical.end(), vertical.begin(), vertical.end());
        else
            std::swap(all_vertical, vertical);
    }


    CVLine middle(cv::Point(0, src_h/2), cv::Point(1, src_h/2));
    std::sort(all_vertical.begin(), all_vertical.end(),
              [&middle](CVLine const &first, CVLine const &second)
    {
        auto res1 = ::line_line_intersection(first.start, first.end, middle.start, middle.end);
        auto res2 = ::line_line_intersection(second.start, second.end, middle.start, middle.end);
        return (res1.first && res2.first) && (res1.second.x < res2.second.x);
    });


//    std::sort(all_vertical.begin(), all_vertical.end(),
//              [&origin](CVLine const &first, CVLine const &second)
//    {
//        return ::line_point_dist(first.start, first.end, origin) <
//        ::line_point_dist(second.start, second.end, origin);
//    });


    auto last = std::unique(all_vertical.begin(), all_vertical.end(), [src_w](CVLine const& first, CVLine const& second){
        return (first == second) || first.very_similar(second, src_w*0.005);
    });

    all_vertical.erase(last, all_vertical.end());
    //cout << all_vertical.size() << endl;

    auto& previous = all_vertical[0];
    for(size_t i = 0; i < all_vertical.size(); ++i)
    {
        //cout << ::RAD_TO_DEGREE(all_vertical[i].inclination - previous.inclination) << endl;
        previous = all_vertical[i];
        Scalar color = colors[i%3];
        cv::line(drawing, all_vertical[i].start, all_vertical[i].end, color, 2);
       // cout << all_vertical[i].length << " " << endl;
    //    cout << ::line_point_dist(all_vertical[i].start, all_vertical[i].end, origin) << endl;
        //auto res1 = ::line_line_intersection(all_vertical[i].start, all_vertical[i].end, middle.start, middle.end);
        imshow(win_name2, drawing);
        //waitKey();
    //    cout << "intersection x = " << res1.second.x << " y = " << res1.second.y << endl;
    }

   // std::vector<float> centers2 = find_best_edges(all_vertical);
    std::vector<float> centers = detect_key_candidates(all_vertical, src.size());
    for(auto& center : centers)
        cv::line(drawing, cv::Point(0, center), cv::Point(src_w-1, center), Scalar(255, 255, 255), 2);
    imshow(win_name2, drawing);
    waitKey();
    vector<float> lengths;
    vector<int> labels1(all_vertical.size());
    for(auto &vert : all_vertical)
        lengths.push_back(vert.length);
    TermCriteria TC(TermCriteria::MAX_ITER | TermCriteria::EPS, 30, 1.0);
    cv::kmeans(Mat(lengths), 4, Mat(labels1), TC, 10, KMEANS_RANDOM_CENTERS);

    //double compactness2 = kmeans(Mat(lengths), 4, Mat(labels2), TC, 30, KMEANS_RANDOM_CENTERS);
    vector<int> label_count(4);
    for(auto lab : labels1)
        ++label_count[lab];

    //if(compactness1 > compactness2)
      //  labels = labels2;

//    for(size_t i = 0; i < labels.size(); ++i)
//    {
//        Scalar color = colors[labels[i]];
//        cv::line(drawing, all_vertical[i].start, all_vertical[i].end, color, 2);
//        cout << all_vertical[i].length << " " << endl;
//        imshow(win_name2, drawing);
//        waitKey();
//    }


    waitKey();
}

static int best_line_fit(CVLine const& line, std::vector<float>& triplet)
{
    std::vector<double> minima(3);
    minima[0] = std::abs(line.start.y - triplet[0]) + std::abs(line.end.y - triplet[2]);
    minima[1] = std::abs(line.start.y - triplet[0]) + std::abs(line.end.y - triplet[1]);
    minima[2] = std::abs(line.start.y - triplet[1]) + std::abs(line.end.y - triplet[2]);
    return std::min_element(minima.begin(), minima.end()) - minima.begin();
}

std::vector<float>  FeatureExtractor::detect_key_candidates(std::vector<CVLine> &vertical_lines, cv::Size sz)
{

    std::vector<float> centers_app(3);
    centers_app = find_best_edges(vertical_lines);
    std::sort(centers_app.begin(), centers_app.end());

    vector<pair<int, int>> key_edges;
    for(size_t i = 0; i < vertical_lines.size(); ++i)
        key_edges.push_back(pair<int, int>(i, best_line_fit(vertical_lines[i], centers_app)));
    for(auto key : key_edges)
        cout << key.second;
    cout << endl;
    cout << endl;
    ///combinations:
    /// 0, 0 - full key
    /// 0, 2, 1 - J key
    /// 1, 2, 1 - T key
    /// 1, 2, 0 - L key
    ///
    std::string keyboard = "";
    std::vector<pair<int, int>> key_vertices;
    int sum = 0;
    for(size_t i = 0; i < key_edges.size(); ++i)
    {
        if(key_edges[i].second == 2)
            continue;

        sum = key_edges[i].second;
        bool spin = true;
        for(size_t j = i + 1; (j < key_edges.size()) && spin; ++j)
        {
            sum = 10*sum + key_edges[j].second;
            cout << sum << endl;
            switch(sum)
            {
               case 0:
                //full key
                keyboard += "O";
                //check valid full key

                key_vertices.push_back(std::pair<int, int>(i, j - i + 1));

                spin = false;
                break;
               case 2:
                //possibly J key
               case 12:
                //possibly T or L key
                break;
               case 21:
                keyboard += "J";
                //check valid J key
                key_vertices.push_back(std::pair<int, int>(i, j - i + 1));
                //J key
                spin = false;
                break;
               case 120:
                //L key
                keyboard += "L";
                //check valid L key
                key_vertices.push_back(std::pair<int, int>(i, j - i + 1));
                spin = false;
                break;
               case 122:
                //possibly T key
                break;
               case 1221:
                keyboard += "T";
                //check valid T key
                key_vertices.push_back(std::pair<int, int>(i, j - i + 1));
                spin = false;
               default:
                spin = false;
                //Error
                break;
            }
        }
    }
    std::cout << std::endl << keyboard << std::endl;

    std::vector<KeyCandidate> kcs;
    for(size_t i = 0; i < keyboard.size(); ++i)
    {
        int st_ind = key_vertices[i].first;
        int num = key_vertices[i].second;
        KeyCandidate KC(std::vector<CVLine>(&vertical_lines[st_ind], &vertical_lines[st_ind+num]), keyboard[i]);
        if(KC.is_valid_candidate())
            kcs.push_back(KC);
    }

    Mat drawing = Mat::zeros(sz, CV_8UC3);
    namedWindow("Win", 0);
    for(auto &kc : kcs)
    {
        Key k(kc.lines_to_polygon());
        k.draw(drawing);

    imshow("Win", drawing);
    waitKey();
    }

    //check if there are more than two repetitions of same line type
    vector<pair<int, int>> repetitions;

    for(size_t i = 0; i < key_edges.size() - 1; ++i)
    {
        int reps = 0;
        for(size_t j = i + 1; j < key_edges.size(); ++j)
        {
            if(key_edges[i].second == key_edges[j].second)
                ++reps;
            else
                break;
        }
        if(reps >= 2)
        repetitions.push_back(pair<int, int>(key_edges[i].first, reps));
    }

    std::vector<int>  indices(vertical_lines.size());
    indices[0] = 0;
    for(size_t i = 1; i < vertical_lines.size(); ++i)
        indices[i] = indices[i-1] + static_cast<int>(
                    ::max_ratio(vertical_lines[i-1].length, vertical_lines[i].length) > 1.1);


    return centers_app;
}

std::vector<float> FeatureExtractor::find_best_edges(std::vector<CVLine> &vertical_lines)
{
    std::vector<float> y_coords;
    for(auto& line : vertical_lines)
    {
        y_coords.push_back(line.start.y);
        y_coords.push_back(line.end.y);
    }
    std::vector<int> labeles(y_coords.size());
    int centers = 2;
    std::vector<float> centers_app;
    double error = INFINITY;
    do
    {
        std::vector<float> centers_app_tmp(++centers);
        double err_tmp = cv::kmeans(Mat(y_coords), centers, Mat(labeles),
                         TermCriteria(TermCriteria::MAX_ITER, 10, 1.0),
                         4, KMEANS_RANDOM_CENTERS, Mat(centers_app_tmp));
        if(err_tmp > error)
            break;
        else
            std::swap(centers_app_tmp, centers_app);
        error = err_tmp;
    }while (centers < 3);

    //find n best labels
    std::vector<int> label_cnt(centers, 0);
    for(auto& labs : labeles)
        ++label_cnt[labs];
    std::vector<float> lines_y;
    for(size_t i = 0; i < 3; ++i)
    {
        auto max_pos = std::max_element(label_cnt.begin(), label_cnt.end());
        lines_y.push_back(centers_app[max_pos - label_cnt.begin()]);
        *max_pos = -1.0;
    }

    return lines_y;
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

    cv::Point origin(-1000, src_sz.height-1);

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
