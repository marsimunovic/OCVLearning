#ifndef DETECTKEYBOARD_H
#define DETECTKEYBOARD_H

#include "opencv2/core/core.hpp"

int detectKeyboard(cv::Mat src, cv::Size pattern);
int histTest(cv::Mat& src);
int histTestC1(cv::Mat& src);
int histTestHUE(cv::Mat& src);
int fourier_analysis(char const* filename);
int edgeDetection(cv::Mat& src);

#endif // DETECTKEYBOARD_H
