#include "keycandidate.h"
#include "math_functions.h"

#include <algorithm>

const int MAX_INCLINATION_DIFF_DEGREES = 6;
const double ALLOWED_RATIO = 1.1;
const double AR = ALLOWED_RATIO;

KeyCandidate::KeyCandidate(const std::vector<CVLine> &lines_, char type)
    :lines(lines_)
    ,type_(type)
{
    if(type == 'B')
    {
        std::vector<CVLine> tmp = {lines[0], lines[2]};
        std::swap(lines, tmp);
    }
}

bool KeyCandidate::is_valid_candidate() const
{
    //first check inclinations
    size_t SZ = lines.size();
    for(size_t i = 0; i < SZ - 1; ++i)
    {
        if(::RAD_TO_DEGREE(lines[i].inclination - lines[i+1].inclination) > MAX_INCLINATION_DIFF_DEGREES)
            return false;
    }
    switch (type_) {
    case 'O':
    case 'B':
        if(SZ == 2)
        {
            //check top and bottom are in same line
            if(::allowed_max_ratio(lines[0].start.y, lines[1].start.y, AR) &&
                ::allowed_max_ratio(lines[0].end.y, lines[1].end.y, AR))
                return true;
        }
        break;
    case 'J':
        if(SZ == 3)
        {
            if(::allowed_max_ratio(lines[0].end.y, lines[1].end.y, AR)  &&
                ::allowed_max_ratio(lines[1].start.y, lines[2].end.y, AR) &&
                ::allowed_max_ratio(lines[0].start.y, lines[2].start.y, AR)
                )
                return true;
        }
        break;
    case 'L':
        if(SZ == 3)
        {
            if(::allowed_max_ratio(lines[0].end.y, lines[1].start.y, AR)  &&
                ::allowed_max_ratio(lines[1].end.y, lines[2].end.y, AR) &&
                ::allowed_max_ratio(lines[0].start.y, lines[2].start.y, AR)
                )
                return true;
        }
        break;
    case 'T':
        if(SZ == 4)
        {
            if(::allowed_max_ratio(lines[0].end.y, lines[1].start.y, AR)  &&
                ::allowed_max_ratio(lines[1].start.y, lines[2].start.y, AR) &&
                ::allowed_max_ratio(lines[1].end.y, lines[2].end.y, AR) &&
                ::allowed_max_ratio(lines[2].start.y, lines[3].end.y, AR)
                )
                return true;
        }
        break;

    default:
        break;
    }
    return false;
}

std::vector<cv::Point> KeyCandidate::lines_to_polygon() const
{
    size_t N = lines.size();
    std::vector<cv::Point> points(N*2);
    points[0] = lines[0].end;
    points[1] = lines[0].start;
    points[2] = lines[N-1].start;
    points[3] = lines[N-1].end;
    if(N > 2)
    {
        if(type_ == 'J' || type_ == 'T')
        {
            points[4] = lines[N-2].start;
            points[5] = lines[N-2].end;
        }
        else
        {
            points[4] = lines[N-2].end;
            points[5] = lines[N-2].start;
        }
    }
    if(N > 3)
    {
        points[6] = lines[N-3].end;
        points[7] = lines[N-3].start;
    }
    return std::move(points);
}

