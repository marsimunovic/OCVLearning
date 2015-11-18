#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>


using namespace cv;
using namespace std;

int maxCorners = 120;
int maxTrackbar = 100;

RNG rng(12345);
char* source_window = "Image";

void goodFeaturesToTrack_Demo(cv::Mat& src)
{
  if( maxCorners < 1 ) { maxCorners = 1; }

  /// Parameters for Shi-Tomasi algorithm
  vector<Point2f> corners;
  double qualityLevel = 0.01;
  double minDistance = 10;
  int blockSize = 3;
  bool useHarrisDetector = false;
  double k = 0.04;

  /// Copy the source image
  Mat copy;
  copy = src.clone();

  Mat src_gray;
  cvtColor( src, src_gray, COLOR_BGR2GRAY );
  /// Apply corner detection
  goodFeaturesToTrack( src_gray,
               corners,
               maxCorners,
               qualityLevel,
               minDistance,
               Mat(),
               blockSize,
               useHarrisDetector,
               k );


  /// Draw corners detected
  cout<<"** Number of corners detected: "<<corners.size()<<endl;
  int r = 4;
  for( int i = 0; i < corners.size(); i++ )
     { circle( copy, corners[i], r, Scalar(rng.uniform(0,255), rng.uniform(0,255),
              rng.uniform(0,255)), -1, 8, 0 ); }

  /// Show what you got
  namedWindow( source_window, WINDOW_AUTOSIZE );
  imshow( source_window, copy );
}

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
        goodFeaturesToTrack_Demo(frameReference);
/*
        Mat dst, cdst;
        Canny(frameReference, dst, 50, 200, 3);

        cvtColor(dst, cdst, COLOR_GRAY2BGR);
        vector<Vec4i> lines;
        HoughLinesP(dst, lines, 1, CV_PI/180,  100, 50, 10 );
        for( size_t i = 1; i < lines.size(); i++ )
        {
            Vec4i l = lines[i];
            Vec4i prev_l = lines[i-1];
            Point a(l[0], l[1]);
            Point b(prev_l[2], prev_l[3]);
            double res = cv::norm(a-b);//Euclidian distance
            cout << i << " : " << res << endl;
            line( frameReference, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
        }

        imshow(WIN_RF, dst);
*/
        int delay = 20000;
        char c = (char)waitKey(delay);
        if(c == 27)
            break;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    readVideo(argc, argv);
    return 0;
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
