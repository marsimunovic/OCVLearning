#include "key.h"
#include <opencv2/imgproc.hpp>


Key::Key()
{
}

Key::Key(std::vector<cv::Point> const& key_coordinates)
    :key_angles(key_coordinates)
    ,tone('C')
    ,octave_num(1)
    ,layout(WHITE_O)
{
}

void Key::press()
{
    pressed = true;
    //TODO: play sound
}

void Key::release()
{
    pressed = false;
    //TODO: release sound
}

void Key::draw(cv::Mat &src_image)
{
    size_t i = 0;
    for(; i < key_angles.size() - 1; ++i)
    {
        cv::line(src_image, key_angles[i], key_angles[i+1], cv::Scalar(255, 255, 255), 2);
        cv::circle(src_image, key_angles[i], 4, cv::Scalar(255, 255, 255), -1, cv::LINE_8);
    }
    cv::circle(src_image, key_angles[i], 4, cv::Scalar(255, 255, 255), -1, cv::LINE_8);
    cv::line(src_image, key_angles[0], key_angles[i], cv::Scalar(255, 255, 255), 2);
}


