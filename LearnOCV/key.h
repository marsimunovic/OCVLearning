#ifndef KEY_H
#define KEY_H



#include <opencv2/core/types.hpp>


class Key
{
public:
    Key();
    Key(const std::vector<cv::Point> &key_coordinates, char type);
    void press();
    void release();
    void draw(cv::Mat& src_image);
private:
    //geometrical shape of key
    enum KEY_LAYOUT
    {
        BLACK,
        WHITE_O, //full white key
        WHITE_L, //key before black doublet
        WHITE_T, //key between black doublet
        WHITE_J  //key after black doublet
    };

private:
    std::vector<cv::Point> key_angles;
    char tone;
    char octave_num;
    bool pressed;
    char layout;
};

typedef std::vector<Key> VKeyboard;

#endif // KEY_H
