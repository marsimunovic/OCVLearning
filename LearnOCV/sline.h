#ifndef SLINE_H
#define SLINE_H

#include "opencv2/core/types.hpp"

void testLine();

class SLine
{
public:
    struct Point
    {
        Point():x(0), y(0){}
        Point(int x_, int y_) : x(x_), y(y_){}
        int x, y;
    };

public:
    SLine();
    SLine(int x1, int y1, int x2, int y2);
    Point start_point() const {return start;}
    Point end_point() const {return end;}

    cv::Point start_cvpt() const {return cv::Point(start.x, start.y);}
    cv::Point end_cvpt() const {return cv::Point(end.x, end.y);}
    Point center_point()  const {return Point((start.x+end.x)/2, (start.y+end.y)/2);}

    static bool compare_by_x(SLine const& first, SLine const &second);
    static bool compare_by_y(SLine const& first, SLine const &second);
    static double calculate_angle(SLine::Point pt1, SLine::Point pt2, SLine::Point pt0);

    bool is_parallel_to(SLine& other_line, int epsilon = 0);
    bool is_vertical_to(SLine& other_line, int epsilon = 0);
    double inclination();
    double norm();
    int dot_product(SLine& other_line);
    void to_upwardy();
    void to_righty();

    double inclination_4quadrant();
private:
    Point start, end;
    inline int dx();
    inline int dy();
};

#endif // SLINE_H
