#ifndef CVLINE_H
#define CVLINE_H

#include "opencv2/core/types.hpp"

struct CVLine
{
    CVLine();
    CVLine(cv::Point const& A, cv::Point const& B);
    CVLine(int x1, int y1, int x2, int y2);
    void shift_right(int offset);
    bool operator==(CVLine const& other) const;
    bool very_similar(const CVLine &other, double eps_distance) const; //experimental function

//members
    cv::Point start;
    cv::Point end;
    float inclination;
    float length;


};

#endif // CVLINE_H
