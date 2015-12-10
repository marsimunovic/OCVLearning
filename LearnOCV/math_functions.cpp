#include "math_functions.h"

int dot(const cv::Point &A, const cv::Point &B, const cv::Point &C)
{
    cv::Point AB = B - A;
    cv::Point BC = C - B;
    int dot = AB.dot(BC);
    return dot;
}

int cross(const cv::Point &A, const cv::Point &B, const cv::Point &C)
{
    cv::Point AB = B - A;
    cv::Point AC = C - A;
    int cross = AB.cross(AC);
    return cross;
}

double distance(const cv::Point &A, const cv::Point &B)
{
    return cv::norm(A - B);
}


double line_point_dist(const cv::Point &A, const cv::Point &B, const cv::Point &C, bool isSegm)
{
    double dist = static_cast<double>(cross(A, B, C))/distance(A,B);
    if(isSegm){
        int dot1 = dot(A, B, C);
        if(dot1 > 0) return distance(B, C);
        int dot2 = dot(B, A, C);
        if(dot2 > 0) return distance(A, C);
    }
    return abs(dist);
}

double max_ratio(double val1, double val2)
{
    return (val1 > val2) ? (val1/val2) : (val2/val1);
}
