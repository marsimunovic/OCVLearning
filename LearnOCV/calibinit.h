#ifndef CALIBINIT_H
#define CALIBINIT_H

#include <opencv2/core/types.hpp>

int cvFindKeyboardCorners(cv::Mat &src, cv::Size size, int flags);

#endif // CALIBINIT_H
