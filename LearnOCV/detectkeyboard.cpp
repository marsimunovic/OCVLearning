#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include <vector>
#include <iostream>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;


Mat sorc, erosion_dst, dilation_dst;
int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

int h = 0, w = 0;

//function headers
void Erosion(int, void *);
void Dilation(int, void *);

int dillate(cv::Mat src, cv::Size size)
{
    //src = imread("../../images/piano_plain1.jpg", 1);
    if(!src.data)
        return -1;

    //namedWindow("Erosion Demo", CV_WINDOW_NORMAL);
    namedWindow("Dilation Demo", CV_WINDOW_NORMAL);
    //MoveWindow("Dilation Demo", src.cols, 0);

    //create erosion trackbar
   /* createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
                      &erosion_elem, max_elem,
                      Erosion );

    createTrackbar( "Kernel size:\n 2n +1", "Erosion Demo",
                      &erosion_size, max_kernel_size,
                      Erosion );
    */
    createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
                      &dilation_elem, max_elem,
                      Dilation );

      createTrackbar( "Kernel size:\n 2n +1", "Dilation Demo",
                      &dilation_size, max_kernel_size,
                      Dilation );
      sorc = src;

      /// Default start
      //Erosion( 0, 0 );
      Dilation( 0, 0 );

      waitKey(0);
    return 0;
}

//EROSION
void Erosion(int, void *)
{
    int erosion_type;
    if(erosion_elem == 0)
        erosion_type = MORPH_RECT;
    else if(erosion_elem == 1)
        erosion_type = MORPH_CROSS;
    else if(erosion_elem == 2)
        erosion_type = MORPH_ELLIPSE;

    Mat element = getStructuringElement(erosion_type,
                                        Size(2*erosion_size + 1, 2*erosion_size + 1),
                                        Point(erosion_size, erosion_size));
    erode(sorc, erosion_dst, element);


    float const black_level = 40.f;
    float const white_level = 180.f;
    float const black_white_gap = 100.f;

    int result = 0;
    Mat thresh;
    for(float thresh_level = black_level; thresh_level < white_level
        && !result; thresh_level += 20.0f)
    {

        cv::threshold(erosion_dst, thresh, thresh_level,
                      255, CV_THRESH_BINARY);
        Mat cdst;
        cvtColor(thresh, cdst, COLOR_GRAY2BGR);
        //vector<vector<Point> > squares;
        //findSquares(thresh, squares);
        //drawSquares(thresh, squares);
        //squares.clear();

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );


        // test each contour
        //Mat drawing = Mat::zeros( thresh.size(), CV_8UC3 );
        for( size_t i = 0; i < contours.size(); i++ )
        {
            // approximate contour with accuracy proportional
            // to the contour perimeter
            //vector<Point> approx;
            //approxPolyDP(Mat(contours[i]), contours[i], arcLength(Mat(contours[i]), true)*0.02, true);
            //std::cout << "Prev size: " << contours[i].size() << " new size: " << approx.size() << std::endl;
            //RNG rng(12345);
            //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( cdst, contours, i, Scalar(0, 255, 0), 2, 8, hierarchy, 0, Point() );
        }

        imshow("Erosion Demo", cdst);
        waitKey(500);
    }


}

//DILATION
void Dilation(int, void *)
{
    int dilation_type;
    if(dilation_elem == 0)
        dilation_type = MORPH_RECT;
    else if(dilation_elem == 1)
        dilation_type = MORPH_CROSS;
    else if(dilation_elem == 2)
        dilation_type = MORPH_ELLIPSE;

    Mat element = getStructuringElement(dilation_type,
                                        Size(2*dilation_size + 1, 2*dilation_size + 1),
                                        Point(dilation_size, dilation_size));
    dilate(sorc, dilation_dst, element);
    Mat dst;



    float const black_level = 40.f;
    float const white_level = 180.f;
    float const black_white_gap = 100.f;

    int result = 0;
    Mat thresh;
    for(float thresh_level = black_level; thresh_level < white_level
        && !result; thresh_level += 20.0f)
    {
        adaptiveThreshold(dilation_dst, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                          THRESH_BINARY, 7, 30);
        //cv::threshold(dilation_dst, thresh, thresh_level + black_white_gap,
          //            255, CV_THRESH_BINARY);
        Mat cdst;
        cvtColor(thresh, cdst, COLOR_GRAY2BGR);
        //vector<vector<Point> > squares;
        //findSquares(thresh, squares);
        //drawSquares(thresh, squares);
        //squares.clear();

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );


        // test each contour
        //Mat drawing = Mat::zeros( thresh.size(), CV_8UC3 );
        for( size_t i = 0; i < contours.size(); i++ )
        {
            // approximate contour with accuracy proportional
            // to the contour perimeter
            //vector<Point> approx;
            //approxPolyDP(Mat(contours[i]), contours[i], arcLength(Mat(contours[i]), true)*0.02, true);
            //std::cout << "Prev size: " << contours[i].size() << " new size: " << approx.size() << std::endl;
            //RNG rng(12345);
            //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( cdst, contours, i, Scalar(0, 255, 0), 2, 8, hierarchy, 0, Point() );
        }

        imshow("Dilation Demo", cdst);
        waitKey(4000);
        break;
    }
}

/*
    import cv2
    import numpy as np

    img = cv2.imread('dave.jpg')
    img = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = np.zeros((gray.shape),np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
    div = np.float32(gray)/(close)
    res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
    res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

    thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
    contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_cnt = None
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = cnt

    cv2.drawContours(mask,[best_cnt],0,255,-1)
    cv2.drawContours(mask,[best_cnt],0,0,2)

    res = cv2.bitwise_and(res,mask)


*/

int detectKeyboard(cv::Mat src, cv::Size pattern)
{

//=====================================================================================

    /// Image PreProcessing
    /** Closing operation*/

    Mat img(src.size(), src.type());
    GaussianBlur(src, img, Size(5,5), 0); //nosie reduction
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(11,11));

    Mat close;
    morphologyEx(gray, close, MORPH_CLOSE, kernel1);
    namedWindow("Step 1", WINDOW_NORMAL);
    Mat img2;
    gray.convertTo(img2, CV_32FC1);
    Mat div;
    Mat img3;
    close.convertTo(img3, CV_32FC1);
    div = img2/img3;

   // for(int i = 0; i < 20; ++i)
     //   std::cout << img2.at<float>(0, i) << " : " << img3.at<float>(0, i) << " = " << div.at<float>(0, i) << std::endl;
    Mat res, res2;
    cv::normalize(div, res, 0, 255, NORM_MINMAX, CV_8UC1);
    //div.convertTo(res, CV_8UC1);
    //for(int i = 0; i < 20; ++i)
      //  std::cout << div.at<float>(0, i) << " : " << static_cast<int>(res.at<uchar>(0, i)) <<  std::endl;
    cvtColor(res, res2, COLOR_GRAY2BGR);
    imshow("Step 1", res);
//=====================================================================================

        /// Finding bounding rectangle of keyboard
        /** Extracting keyboard mask*/

    Mat thresh;
    adaptiveThreshold(res, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 2);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    double max_area = 0.0;
    size_t best_cnt = 0;
    for(size_t i = 0; i < contours.size(); ++i)
    {
        double area = contourArea(contours[i]);
        if(area > 400.0)
        {
            if(area > max_area)
            {
                max_area = area;
                best_cnt = i;
            }
        }
    }

    Mat mask = Mat::zeros( gray.size(), CV_8UC1);
    drawContours(mask, contours, best_cnt, 255, -1); // contour + fill
    drawContours(mask, contours, best_cnt, 0, 2); //erase contour and leave fill
    bitwise_and(res, mask, res);
    namedWindow("Step 2", WINDOW_NORMAL);
   // moveWindow("Step 2", 0, res.size().height/2);
    imshow("Step 2", res);
    //waitKey(0);

//=====================================================================================

        /// Finding vertical lines
        /** */

/*
kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

dx = cv2.Sobel(res,cv2.CV_16S,1,0)
dx = cv2.convertScaleAbs(dx)
cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if h/w > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)
close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
closex = close.copy()
*/
    Mat kernelx = getStructuringElement(MORPH_RECT, Size(2,10));

    Mat dx;
    Sobel(res, dx, CV_16S, 1, 0);
    convertScaleAbs(dx, dx);
    normalize(dx, dx, 0, 255, NORM_MINMAX);
    Mat closex;
    double retx = threshold(dx, closex, 0, 255, THRESH_BINARY | THRESH_OTSU);
    morphologyEx(closex, closex, MORPH_DILATE, kernelx);
    contours.clear();
    hierarchy.clear();
    findContours(closex, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for(size_t i = 0; i < contours.size(); ++i)
    {
        Rect R = boundingRect(contours[i]);
        if(R.height/R.width > 5)
            drawContours(closex, contours, i, 255, -1);
        else
            drawContours(closex, contours, i, 0, -1);
    }
    //Mat dilx = getStructuringElement(MORPH_RECT, Size(3,3));
    //dilate(closex, closex, dilx , Point(-1, -1), 2);
    namedWindow("Step 3", WINDOW_NORMAL);
    imshow("Step 3", closex);

//=====================================================================================

    /// Finding horizontal lines
            /** */

    Mat kernely = getStructuringElement(MORPH_RECT, Size(10,2));
    Mat dy;
    Sobel(res, dy, CV_16S, 0, 2);
    convertScaleAbs(dy, dy);
    normalize(dy, dy, 0, 255, NORM_MINMAX);
    Mat closey;
    double rety = threshold(dy, closey, 0, 255, THRESH_BINARY | THRESH_OTSU);
    morphologyEx(closey, closey, MORPH_DILATE, kernely);
    contours.clear();
    hierarchy.clear();
    findContours(closey, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for(size_t i = 0; i < contours.size(); ++i)
    {
        Rect R = boundingRect(contours[i]);
        if(R.width/R.height > 5)
            drawContours(closey, contours, i, 255, -1);
        else
            drawContours(closey, contours, i, 0, -1);
    }
    Mat dily = getStructuringElement(MORPH_RECT, Size(3,3));
    dilate(closey, closey, dily , Point(-1, -1), 2);
    namedWindow("Step 4", WINDOW_NORMAL);
    imshow("Step 4", closey);

//=====================================================================================

    /// Finding intersection points
    /** */

    Mat intersection;
    bitwise_and(closex, closey, intersection);
    namedWindow("Step 5", WINDOW_NORMAL);
    imshow("Step 5", intersection);

//=====================================================================================

    /// Finding intersection point moments
    /** */

    contours.clear();
    hierarchy.clear();
    findContours(intersection, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    vector<std::pair<int, int> > centroids;



    for(size_t i = 0; i < contours.size(); ++i)
    {
        Moments mom = moments(contours[i]);
        int x = static_cast<int>(mom.m10/mom.m00);
        int y = static_cast<int>(mom.m01/mom.m00);
        circle(img, Point(x,y), 4, Scalar(0,255,0), -1);
        //putText(img, std::to_string(i), Point(x, y), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0));
        centroids.push_back(std::pair<int, int>(x,y));
    }
    cout << norm(Point(3, 4)) << endl;


    stable_sort(centroids.begin(), centroids.end(),
         [](pair<int, int> const& a, pair<int, int> const& b) -> bool
        {

            return (a.second) < (b.second);
              });

    for(int i = 0; i < 100; i+=10)
    {
    stable_sort(centroids.begin() + i, centroids.begin() + i + 10,
         [](pair<int, int> const& a, pair<int, int> const& b) -> bool
        {
            return a.first < b.first;
        });
    }

    for(size_t i = 0; i < centroids.size(); ++i)
    {
        int x_ = std::get<0>(centroids[i]);
        int y_ = std::get<1>(centroids[i]);
        putText(img, std::to_string(i), Point(x_, y_), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0));
        cout << i << " : (" <<to_string(x_) << "," << to_string(y_) << ")" << endl;
    }
    namedWindow("Step 6", WINDOW_NORMAL);
    imshow("Step 6", img);

//=====================================================================================

    /// Correcting image
    /** */
    Mat output = Mat::zeros(450, 450, CV_8UC3);

    namedWindow("Step 7", WINDOW_NORMAL);
    for(size_t i = 0; i < centroids.size(); ++i)
    {
        int ri = i/10;
        int ci = i%10;
        if(ci != 9 && ri != 9)
        {
            std::vector<Point2f> src_corners(4);
            src_corners[0] = Point(centroids[i].first, centroids[i].second);
            src_corners[1] = Point(centroids[i+1].first, centroids[i+1].second);
            src_corners[2] = Point(centroids[i+10].first, centroids[i+10].second);
            src_corners[3] = Point(centroids[i+11].first, centroids[i+11].second);

            std::vector<Point2f> dst_corners(4);
            dst_corners[0] = Point(ci*50, ri*50);
            dst_corners[1] = Point((ci+1)*50-1, ri*50);
            dst_corners[2] = Point(ci*50, (ri+1)*50-1);
            dst_corners[3] = Point((ci+1)*50-1, (ri+1)*50-1);
            Mat retval = getPerspectiveTransform(src_corners, dst_corners);
            Mat roi(output, Rect(ci*50, ri*50, 50, 50));
            Mat tmp;
            warpPerspective(res2, tmp, retval, Size(450, 450));
            Mat roi2(tmp, Rect(ci*50, ri*50, 50, 50));
            roi2.copyTo(roi);
  //          imshow("Step 7", output);
    //        waitKey(0);
        }

    }
    imshow("Step 7", output);


    waitKey(0);
    return 0;
}


