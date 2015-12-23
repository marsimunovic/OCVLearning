#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

#include "opencv2/core/types.hpp"

double DEGREE_TO_RAD(double degree);
double RAD_TO_DEGREE(double rad);

int dot(cv::Point const& A, cv::Point const& B, cv::Point const& C);
int cross(cv::Point const& A, cv::Point const& B, cv::Point const& C);
double distance(cv::Point const& A, cv::Point const& B);
double line_point_dist(cv::Point const& A, cv::Point const& B,
                       cv::Point const& C, bool isSegm = false);

std::pair<bool, cv::Point> line_line_intersection(
        cv::Point const& A, cv::Point const& B, cv::Point const& C, cv::Point const& D);

double max_ratio(double const val1, double const val2);
bool max_ratio(double const val1, double const val2, double &ratio);
bool allowed_max_ratio(double const val1, double const val2, double const allowed);

double line_inclination(cv::Point const &A, cv::Point const &B);
cv::Point center_of_simmetry(cv::Point const &A, cv::Point const &B);

#endif // MATH_FUNCTIONS_H
