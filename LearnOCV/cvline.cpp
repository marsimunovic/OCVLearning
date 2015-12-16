#include "cvline.h"
#include "math_functions.h"

CVLine::CVLine()
    :start(0,0)
    ,end(0,0)
    ,inclination(0.f)
    ,length(0.f)
{
}

CVLine::CVLine(const cv::Point &A, const cv::Point &B)
    :start(A)
    ,end(B)
    ,inclination(::line_inclination(A, B))
    ,length(::distance(A, B))
{
    if(A.y > B.y)
        std::swap(start, end);
}

CVLine::CVLine(int x1, int y1, int x2, int y2)
    :start(x1, y1)
    ,end(cv::Point(x2, y2))
    ,inclination(::line_inclination(start, end))
    ,length(::distance(start, end))
{
    if(start.y > end.y)
        std::swap(start, end);
}


void CVLine::shift_right(int offset)
{
    start = start + cv::Point(offset, 0);
    end = end + cv::Point(offset, 0);
}

bool CVLine::operator==(const CVLine &other) const
{
    return ((start.x == other.start.x) && (start.y == other.start.y) &&
            (end.x == other.end.x) && (end.y == other.end.y));
}

bool CVLine::very_similar(CVLine const& other, double eps_distance) const
{
    if(std::abs(inclination - other.inclination) > DEGREE_TO_RAD(5))
        return false;
    if(::max_ratio(length, other.length) > 1.1)
        return false;
    if(::distance(center_of_simmetry(start, end), center_of_simmetry(other.start, other.end)) > eps_distance)
        return false;
    return true;
}
