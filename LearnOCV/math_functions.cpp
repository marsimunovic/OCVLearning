#include "math_functions.h"
#include <cmath>

//calculates dot product of vectors from A to B and from B to C
int dot(const cv::Point &A, const cv::Point &B, const cv::Point &C)
{
    cv::Point AB = B - A;
    cv::Point BC = C - B;
    int dot = AB.dot(BC);
    return dot;
}


//calculates cross product of vectors from A to B and from A to C
int cross(const cv::Point &A, const cv::Point &B, const cv::Point &C)
{
    cv::Point AB = B - A;
    cv::Point AC = C - A;
    int cross = AB.cross(AC);
    return cross;
}

//calculates distance from point A to B
double distance(const cv::Point &A, const cv::Point &B)
{
    return cv::norm(A - B);
}


//calculates smallest distance from line AB to point C (vertical distance)
//if isSegm true then calculate distance line segment AB
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

//maximized ratio between two values
double max_ratio(double val1, double val2)
{
    return (val1 > val2) ? (val1/val2) : (val2/val1);
}

//maximized ratio between two values
//returns true if val1 is bigger than val2
bool max_ratio(double val1, double val2, double &ratio)
{
    if(val1 > val2)
    {
        ratio = val1/val2;
        return true;
    }
    else
    {
        ratio = val2/val1;
        return false;
    }
}

double line_inclination(cv::Point const &A, cv::Point const &B)
{
    cv::Point delta = B - A;
    abs(atan(abs(delta.y/(delta.x + 0.00000000001))));
}
