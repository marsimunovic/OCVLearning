#include "detectkeys.h"
#include "sline.h"

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

int calc_treshold(cv::Mat& src)
{
    Mat dst, bw_hist;
    //cvtColor(src, hsv, CV_BGR2HSV);
    //cvtColor(src, dst, CV_8UC1);
    int histSize = 256; //number of bins
    float range[] = {0, 256};
    const float* histRange = {range};
    int channs[] = {0};

    calcHist(&src, 1, channs, Mat(), bw_hist, 1, &histSize, &histRange);
    int min_ind = 130;
    int max_ind = 180;
    float min_val = static_cast<float>(src.size().area());
    //find best position for mode separation
    //TODO: determine mode ranges instead of fixed min_ind and max_ind
    for(int i = min_ind; i < max_ind; ++i)
    {
        if(min_val >= bw_hist.at<float>(i))
        {
            //cout << i << " : " << bw_hist.at<float>(i) << endl;
            min_val = bw_hist.at<float>(i);
            min_ind = i;
        }
    }
/*
    int hist_w = 256; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
    normalize(bw_hist, bw_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(bw_hist.at<float>(i-1)) ) ,
                         Point( bin_w*(i), hist_h - cvRound(bw_hist.at<float>(i)) ),
                         Scalar( 255, 0, 0), 2, 8, 0  );
        line(histImage, Point(min_ind, 0), Point(min_ind, 399), Scalar(255, 0, 0), 2);

    }

    namedWindow("calcHist Demo", 0 );
    imshow("calcHist Demo", histImage );
*/
    return min_ind;
}


void detect_contours(cv::Mat& src, vector<vector<Point>>& approx_contours)
{
    vector<vector<Point>> contours;
    Mat closex = Mat::zeros(src.size(), CV_8UC3);
     Mat brief = Mat::zeros(src.size(), CV_8UC3);
    findContours(src, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    //findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    namedWindow("Contours", WINDOW_NORMAL);
    namedWindow("Brief", WINDOW_NORMAL);
    size_t min_len = src.size().width/15;
    Scalar colors[3] = {Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255)};
    int cntr = 0;
    int brcntr = 0;
    /// Find the convex hull object for each contour
//      vector<vector<Point> >hull( contours.size() );
//      for( int i = 0; i < contours.size(); i++ )
//         {  convexHull( Mat(contours[i]), hull[i], false ); }


//      /// Draw contours + hull results
//      namedWindow( "Hull demo", 0 );
//      Mat drawing = Mat::zeros( src.size(), CV_8UC3 );
//      for( int i = 0; i< contours.size(); i++ )
//         {
//          if(contours[i].size() < min_len)
//                          continue;
//           cntr = (cntr+1) % 3;
//           Scalar color = colors[cntr];
//           drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
//           imshow( "Hull demo", drawing );
//           waitKey(0);
//           cntr = (cntr+1) % 3;
//           color = colors[cntr];
//           drawContours( drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
//           /// Show in a window
//           imshow( "Hull demo", drawing );
//           waitKey(0);
//         }



    for(size_t i = 0; i < contours.size(); ++i)
    {

            if(contours[i].size() < min_len)
                continue;
            vector<Point> brief_contour;
            //double epsilon = arcLength(Mat(contours[i]), true)*0.01;
            approxPolyDP(contours[i], brief_contour, 20, false);
            for(size_t j = 1; j < brief_contour.size(); ++j)
            {
                Point start = brief_contour[j-1];
                Point end = brief_contour[j];
                line(brief, start, end, colors[brcntr], 2);
                brcntr = (brcntr+1) % 3;
            }
            approx_contours.push_back(std::move(brief_contour));

            drawContours(closex, contours, i, colors[cntr], 2);

            cntr = (cntr+1) % 3;
//            imshow("Brief", brief);
//            imshow("Contours", closex);
//            waitKey();
    }
    //Mat dilx = getStructuringElement(MORPH_RECT, Size(3,3));
    //dilate(closex, closex, dilx , Point(-1, -1), 2);

}

int detect_key_candidates(cv::Mat &src, vector<vector<Point>>& approx_contours)
{
    std::vector<SLine> vertical_lines;
    std::vector<SLine> horizontal_lines;

    for(auto& approx_contour : approx_contours)
    {
        for(size_t i = 1; i < approx_contour.size(); ++i)
        {
            Point start = approx_contour[i-1];
            Point end = approx_contour[i];
            SLine l(start.x, start.y, end.x, end.y);
            if(l.inclination() > (M_PI/6))
            {
                //verticalish
                l.to_upwardy();
                vertical_lines.push_back(std::move(l));
            }
            else
            {
                //horizontalish
                l.to_righty();
                horizontal_lines.push_back(std::move(l));
            }
        }
    }

    std::cout << "start sorting " << endl;
    std::sort(vertical_lines.begin(), vertical_lines.end(),
              [](SLine const& first, SLine const& second)
              {
                return SLine::compare_by_x(first, second);
              });
    std::sort(horizontal_lines.begin(), horizontal_lines.end(),
              [](SLine const& first, SLine const& second)
              {
                return SLine::compare_by_y(first, second);
              });

    Mat drawing = Mat::zeros(src.size(), src.type());

    namedWindow("VerticalHorizontal", 0);
    for(auto &vertical_line : vertical_lines)
    {
        SLine::Point pt1 = vertical_line.start_point();
        SLine::Point pt2 = vertical_line.end_point();
        cv::line(drawing, Point(pt1.x, pt1.y), Point(pt2.x, pt2.y), 255, 2);
        cout << "x1 = " << pt1.x << " y1 = " << pt1.y << "x2 = " << pt2.x << " y2 = " << pt2.y << endl;
        imshow("VerticalHorizontal", drawing);
        waitKey(0);
    }
    return 0;
}

int detectKeys(cv::Mat& src)
{

    int const n_src_parts = 4;
    //split image in 4 parts of equal width

    int src_h = src.size().height;
    int src_w = src.size().width;
    int segment_w = src_w/n_src_parts;
    //last segment width my differ if number is not divisible with 4
    int last_segment_w = src_w - (n_src_parts-1)*segment_w;
    cv::Mat quarter[n_src_parts];

    std::string win_name = "Segment ";
    //each segment is processed separately in order
    //to get better local parameters and reduce
    //camera position influence
    cv::Mat src_C1(src.size(), CV_8UC1);
    GaussianBlur(src, src, Size(5, 5), 3, 3);
    cvtColor(src, src_C1, COLOR_BGR2GRAY, 1);
    cout << src_C1.type() << endl;
    for(int i = 0; i < n_src_parts; ++i)
    {

        quarter[i] = cv::Mat(src_C1, Rect(i*segment_w, 0,
                     (i < n_src_parts -1)? segment_w : last_segment_w, src_h));
        //determine image "energy distribution" in order to
        //get the best possible tresholding

        int thresh_val = calc_treshold(quarter[i]);
        cout << "thresh value " << thresh_val << endl;

        cv::Mat bw_quarter[n_src_parts];

        threshold(quarter[i], bw_quarter[i], thresh_val, 255, THRESH_BINARY);

        //perform analysis of black-n-white segment
        //in order to extract geometrical shapes
        vector<vector<Point>> approx_contours;
        detect_contours(bw_quarter[i], approx_contours);

        //use detected lines to detect possible keys
        detect_key_candidates(bw_quarter[i], approx_contours);

        namedWindow(win_name + char(i + 49), 0);
        imshow(win_name + char(i + 49), bw_quarter[i]);
        waitKey();
    }
    return 0;


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


