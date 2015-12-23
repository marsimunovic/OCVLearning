#ifndef KEYCANDIDATE_H
#define KEYCANDIDATE_H

#include "cvline.h"

class KeyCandidate
{
public:

    KeyCandidate(std::vector<CVLine> const& lines, char type);
    bool is_valid_candidate();
    std::vector<cv::Point> lines_to_polygon();
private:
    std::vector<CVLine> lines;
    char type;
};

#endif // KEYCANDIDATE_H
