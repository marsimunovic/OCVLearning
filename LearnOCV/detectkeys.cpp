#include "detectkeys.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include <vector>
#include <iostream>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;


void createBWImage(cv::Mat& src, cv::Mat& img)
{
    GaussianBlur(src, img, Size(5,5), 5); //nosie reduction
    //cv::addWeighted(src, 1.5, img, -0.5, 0, img);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat thresh(img.size(), CV_8UC1);
    threshold(gray, thresh, 145, 255, THRESH_BINARY);
    Mat kernelxy = getStructuringElement(MORPH_RECT, Size(3,5));
    morphologyEx(thresh, img, MORPH_DILATE, kernelxy, Point(-1, -1), 1);
    morphologyEx(thresh, thresh, MORPH_ERODE, kernelxy, Point(-1, -1), 4);
    pyrDown(img, img);
}

void cornerHarris_demo(cv::Mat& src)
{
  string const corners_window = "Corners window";
  Mat dst, dst_norm, dst_norm_scaled;
  dst = Mat::zeros( src.size(), CV_32FC1 );

  /// Detector parameters
  int blockSize = 2;
  int apertureSize = 3;
  double k = 0.04;
  int thresh = 200;

  /// Detecting corners
  cornerHarris( src, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

  /// Normalizing
  normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  convertScaleAbs( dst_norm, dst_norm_scaled );

  /// Drawing a circle around corners
  for( int j = 0; j < dst_norm.rows ; j++ )
     { for( int i = 0; i < dst_norm.cols; i++ )
          {
            if( (int) dst_norm.at<float>(j,i) > thresh )
              {
               circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
              }
          }
     }
  /// Showing the result
  namedWindow( corners_window, CV_WINDOW_NORMAL);
  imshow( corners_window, dst_norm_scaled );
}


void BlobDetector(cv::Mat& src)
{
    Size dims = src.size();
    int h = dims.height;
    int w = dims.width;
    //randomly create search points
    RNG rng(12345);


    //create ROIs around search points

    //find bounded white objects
    //and create a list of results for further processing

}


void findVerticalSegments(cv::Mat& res, cv::Mat& closex)
{
    Mat kernelx = getStructuringElement(MORPH_RECT, Size(2,10));

    Mat dx;
    Sobel(res, dx, CV_16S, 1, 0);
    convertScaleAbs(dx, dx);
    normalize(dx, dx, 0, 255, NORM_MINMAX);
    double retx = threshold(dx, closex, 0, 255, THRESH_BINARY | THRESH_OTSU);
    //morphologyEx(closex, closex, MORPH_DILATE, kernelx);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(closex, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for(size_t i = 0; i < contours.size(); ++i)
    {
        Rect R = boundingRect(contours[i]);
        if(R.height/R.width > 2)
            drawContours(closex, contours, i, 255, -1);
        else
            drawContours(closex, contours, i, 0, -1);
    }
    //Mat dilx = getStructuringElement(MORPH_RECT, Size(3,3));
    //dilate(closex, closex, dilx , Point(-1, -1), 2);
    namedWindow("Step 3", WINDOW_NORMAL);
    imshow("Step 3", closex);
}
//=====================================================================================

    /// Finding horizontal lines
            /** */

void findHorizontalSegments(cv::Mat& res, cv::Mat& closey)
{
    Mat kernely = getStructuringElement(MORPH_RECT, Size(10,2));
    Mat dy;
    Sobel(res, dy, CV_16S, 0, 2);
    convertScaleAbs(dy, dy);
    normalize(dy, dy, 0, 255, NORM_MINMAX);
    double rety = threshold(dy, closey, 0, 255, THRESH_BINARY | THRESH_OTSU);
   // morphologyEx(closey, closey, MORPH_DILATE, kernely);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(closey, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for(size_t i = 0; i < contours.size(); ++i)
    {
        Rect R = boundingRect(contours[i]);
        if(R.width/R.height > 3)
            drawContours(closey, contours, i, 255, -1);
        else
            drawContours(closey, contours, i, 0, -1);
    }
    Mat dily = getStructuringElement(MORPH_RECT, Size(3,3));
    dilate(closey, closey, dily , Point(-1, -1), 2);
    namedWindow("Step 4", WINDOW_NORMAL);
    imshow("Step 4", closey);
    waitKey(0);

}

void findIntersectionPoints(cv::Mat& closex, cv::Mat& closey)
{
    Mat intersection;;
    bitwise_and(closex, closey, intersection);
    namedWindow("Step 5", WINDOW_NORMAL);
    imshow("Step 5", intersection);
    waitKey(0);
}


void calculateHough(cv::Mat& src)
{

    Mat dst, cdst;
     Canny(src, dst, 50, 200, 3);
     cvtColor(dst, cdst, CV_GRAY2BGR);

     #if 0
      vector<Vec2f> lines;
      HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );

      for( size_t i = 0; i < lines.size(); i++ )
      {
         float rho = lines[i][0], theta = lines[i][1];
         Point pt1, pt2;
         double a = cos(theta), b = sin(theta);
         double x0 = a*rho, y0 = b*rho;
         pt1.x = cvRound(x0 + 1000*(-b));
         pt1.y = cvRound(y0 + 1000*(a));
         pt2.x = cvRound(x0 - 1000*(-b));
         pt2.y = cvRound(y0 - 1000*(a));
         line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
      }
     #else
      vector<Vec4i> lines;
      HoughLinesP(dst, lines, 1, CV_PI/180, 30, 50, 50 );

      for( size_t i = 0; i < lines.size(); i++ )
      {
        Vec4i l = lines[i];
        line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,255,0), 3, CV_AA);
      }
     #endif
     namedWindow("detected lines", 0);
     imshow("detected lines", cdst);

     waitKey();

}

int detectKeys(cv::Mat& src)
{


    Mat img(src.size(), src.type());
    createBWImage(src, img);
    calculateHough(img);
//    //cornerHarris_demo(img);
//    Mat vertical(src.size(), src.type());
//    findVerticalSegments(img, vertical);
//    Mat horizontal(src.size(), src.type());
//    findHorizontalSegments(img, horizontal);

//    Mat intersection(src.size(), src.type());
//    findIntersectionPoints(vertical, horizontal);
//    Mat curves;
//    Canny(thresh, curves, 110, 200);
//    vector<Vec4i> lines;
//    HoughLinesP(curves, lines, 1, CV_PI/180,  155, 30, 10 );
//    Mat frameReference = Mat::zeros(thresh.size(), CV_8UC3);
//    for( size_t i = 1; i < lines.size(); i++ )
//    {
//        Vec4i l = lines[i];
//        Vec4i prev_l = lines[i-1];
//        Point a(l[0], l[1]);
//        Point b(prev_l[2], prev_l[3]);
//        double res = cv::norm(a-b);//Euclidian distance
//        cout << i << " : " << res << endl;
//        line( frameReference, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
//    }

    namedWindow("Sharpen", 0);
    imshow("Sharpen", img);
    waitKey(0);

    return 0;
}


