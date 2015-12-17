#include "math_functions.h"
#include <cmath>


double RAD_TO_DEGREE(double rad)
{
    return 180.0*rad/M_PI;
}

double DEGREE_TO_RAD(double degree)
{
    return degree*M_PI/180.0;
}

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
    return std::abs(atan(std::abs(delta.y/(delta.x + 0.00000000001))));
}

cv::Point center_of_simmetry(cv::Point const &A, cv::Point const &B)
{
    return cv::Point((A.x+B.x)/2, (A.y+B.y)/2);
}


std::pair<bool, cv::Point> line_line_intersection(const cv::Point &A, const cv::Point &B,
                                                  const cv::Point &C, const cv::Point &D)
{
    int dy1 = B.y - A.y;
    int dx1 = A.x - B.x;
    int dy2 = D.y - C.y;
    int dx2 = C.x - D.x;
    int z1 = dy1*A.x + dx1*A.y;
    int z2 = dy2*C.x + dx2*C.y;

    int det = dy1*dx2 - dy2*dx1;

    if(det == 0)
        return std::pair<bool, cv::Point>(false, cv::Point(0,0));
    else
    {
        double x = static_cast<double>(dx2*z1-dx1*z2)/det;
        double y = static_cast<double>(dy1*z2-dy2*z1)/det;
        return std::pair<bool, cv::Point>(true, cv::Point(x, y));
    }
}
