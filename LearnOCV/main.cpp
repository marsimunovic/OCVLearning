#include "detectkeyboard.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{
    //string imageName("../data/images/piano_plain1.jpg");
    string imageName("../data/images/sudoku.png");
    //string imageName("../data/images/keyboard.jpg");
  // string imageName("../data/images/piano.png");

    if(argc > 1)
    {
        imageName = argv[1];
    }

    Mat image;
    image = imread(imageName.c_str(), IMREAD_COLOR); //read the file
    detectKeyboard(image, Size(8,5));

    return 0;
}
