#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/highgui.hpp>

#include <stdarg.h>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

static int cvCheckKeyboard(cv::Mat &src, cv::Mat& orig, cv::Size size);


int thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 1; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                        squares.push_back(approx);
                }
            }
        }
    }
}


// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        Mat cimage;
        cvtColor(image, cimage, CV_GRAY2BGR);
        polylines(cimage, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }
    namedWindow(wndname, WINDOW_NORMAL);
    imshow(wndname, image);
    waitKey(0);
}



//=====================================================================================

/// Corner info structure
/** This structure stores information about the keyboard corner.*/
struct CvKBCorner
{
    CvPoint2D32f pt;    //coordinates of the corner
    int row;            //Board row index
    int count;          //Number of neighbor corners
    struct CvKBCorner *neighbors[4]; //Neighbor corners

    float meanDist(int *n_) const
    {
        float sum = 0;
        int n = 0;
        for(int i = 0; i < 4; ++i)
        {
            if(neighbors[i])
            {
                float dx = neighbors[i]->pt.x - pt.x;
                float dy = neighbors[i]->pt.y - pt.y;
                sum += sqrt(dx*dx + dy*dy);
                ++n;
            }
        }
        if(n_)
            *n_ = n;
        return sum/MAX(n,1);
    }
};

int cvFindKeyboardCorners(cv::Mat &src, cv::Size pattern_size,
                          //CvPoint2D32f *out_corners, int* out_corner_count,
                          int flags)
{
    int found = 0;
    int const min_dilations = 0;
    int const max_dilations = 7;


    if(pattern_size.width < 8 || pattern_size.height < 5)
        CV_Error(CV_StsOutOfRange, "There should be at least 8 white and 5 black keys!");

    //if(!out_corners)
      //  CV_Error(CV_StsNullPtr, "Null pointer to corners");

    Mat tresh_img(src.size(), CV_8UC1);
    Mat norm_img(src.size(), CV_8UC1);
    Mat src_orig = src;

    if(src.channels() != 1 || (flags & CV_CALIB_CB_NORMALIZE_IMAGE))
    {
        //equalize input image histogram -
        //that should make the contrast between "black" and "white" areas big enough
        if(CV_MAT_CN(src.type()) != 1)
        {
            cv::cvtColor(src, norm_img, CV_BGR2GRAY);
            src = norm_img;
        }
        if(flags & CV_CALIB_CB_NORMALIZE_IMAGE)
        {
            cv::equalizeHist(src, norm_img);
            src = norm_img;
        }
    }


    cvCheckKeyboard(src, src_orig, Size());


    return found;
}

//=====================================================================================
/// checks if keyboard is in image
/// - src : input image
/// - size : keyboard structure (black and white count)
/// returns 1 if a keyboar can be in this image, 0 if there is no keyboard
/// or -1 in case of error

int cvCheckKeyboard(cv::Mat &src, cv::Mat &orig, cv::Size size)
{
    if(src.channels() > 1)
    {
        cvError(CV_BadNumChannels, "cvCheckKeyboard", "supports single-channel images only",
                __FILE__, __LINE__);
    }

    //TODO: check image depth

    int const erosion_count = 1;
    float const black_level = 20.f;
    float const white_level = 130.f;
    float const black_white_gap = 70.f;

    Mat white = src.clone();
    Mat black = src.clone();
    IplImage tmp_w = white;
    cvErode(&tmp_w, &tmp_w, NULL, erosion_count);
    IplImage tmp_b = black;
    cvDilate(&tmp_b, &tmp_b, NULL, erosion_count);

    Mat thresh(src.size(), IPL_DEPTH_8U, 1);

    int result = 0;
    for(float thresh_level = black_level; thresh_level < white_level
        && !result; thresh_level += 20.0f)
    {
        cv::threshold(white, thresh, thresh_level + black_white_gap,
                      255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        //vector<vector<Point> > squares;
        //findSquares(thresh, squares);
        //drawSquares(thresh, squares);
        //squares.clear();

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );


        // test each contour
        Mat drawing = Mat::zeros( thresh.size(), CV_8UC3 );
        for( size_t i = 0; i < contours.size(); i++ )
        {
            // approximate contour with accuracy proportional
            // to the contour perimeter
            vector<Point> approx;
            //approxPolyDP(Mat(contours[i]), contours[i], arcLength(Mat(contours[i]), true)*0.01, true);
            //std::cout << "Prev size: " << contours[i].size() << " new size: " << approx.size() << std::endl;
            RNG rng(12345);
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
             drawContours( src, contours, i, color, 2, 8, hierarchy, 0, Point() );
           }
        namedWindow("1", WINDOW_NORMAL);
        imshow("1", src);
        waitKey(0);

        cv::threshold(black, thresh, thresh_level, 255, CV_THRESH_BINARY_INV);
        findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
        /// Draw contours
        ///
        ///
        ///
          ///Mat drawing = Mat::zeros( thresh.size(), CV_8UC3 );
        /*
          RNG rng(12345);
          for( int i = 0; i< contours.size(); i++ )
             {
               Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
               drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
             }
        */

    }


    return 0;
}
