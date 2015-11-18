#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>


using namespace cv;
using namespace std;

int readVideo(int argc, char *argv[])
{
    stringstream conv;

    string sourceReference = "../data/video/piano_simple1.mp4";
    if(argc > 2)
    {
        sourceReference = argv[2];
    }
    int frameNum = -1;

    VideoCapture captRefrnc(sourceReference);

    if(!captRefrnc.isOpened())
    {
        cout << "Could not open video reference " << sourceReference << endl;
        return -1;
    }

    Size refS = Size((int) captRefrnc.get(CAP_PROP_FRAME_WIDTH),
                     (int) captRefrnc.get(CAP_PROP_FRAME_HEIGHT));

    cout << "Reference frame resolution: Width=" << refS.width << "  Height=" << refS.height
             << " of nr#: " << captRefrnc.get(CAP_PROP_FRAME_COUNT) << endl;
    char const *WIN_RF = "Reference";

    namedWindow(WIN_RF, WINDOW_AUTOSIZE);

    Mat frameReference;

    for(;;)
    {
        captRefrnc >> frameReference;

        if(frameReference.empty())
        {
            cout << " < < < Video over! > > > " << endl;
            break;
        }
        ++frameNum;
        if(frameNum%4)
            continue;
        cout << "Frame: " << frameNum << "# " << endl;

        Mat dst, cdst;
        Canny(frameReference, dst, 50, 200, 3);
        cvtColor(dst, cdst, COLOR_GRAY2BGR);
        vector<Vec4i> lines;
        HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i l = lines[i];
            line( frameReference, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
        }


        imshow(WIN_RF, frameReference);
        int delay = 20;
        char c = (char)waitKey(delay);
        if(c == 27)
            break;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    //readVideo(argc, argv);
    //return 0;
    string imageName("../data/images/piano_plain1.jpg");
    if(argc > 1)
    {
        imageName = argv[1];
    }

    Mat image;
    image = imread(imageName.c_str(), IMREAD_COLOR); //read the file

    if(image.empty())
    {
        cout << "Could not open or find the image " << imageName << endl;
        return -1;
    }

    namedWindow("Display window", WINDOW_NORMAL); //Create a window
    imshow("Display window", image);


    waitKey(0);
    return 0;
}
