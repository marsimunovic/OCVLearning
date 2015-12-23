#ifndef KEYCANDIDATE_H
#define KEYCANDIDATE_H

#include "cvline.h"

class KeyCandidate
{
public:

    KeyCandidate(std::vector<CVLine> const& lines_, char type);
    bool is_valid_candidate() const;
    std::vector<cv::Point> lines_to_polygon() const;
    inline char type() const { return type_;}
private:
    std::vector<CVLine> lines;
    char type_;
};

#endif // KEYCANDIDATE_H
