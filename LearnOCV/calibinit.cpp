#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/highgui.hpp>

#include <stdarg.h>


using namespace cv;

static int cvCheckKeyboard(cv::Mat &src, cv::Size size);

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


    cvCheckKeyboard(src, Size());


    return found;
}

//=====================================================================================
/// checks if keyboard is in image
/// - src : input image
/// - size : keyboard structure (black and white count)
/// returns 1 if a keyboar can be in this image, 0 if there is no keyboard
/// or -1 in case of error

int cvCheckKeyboard(cv::Mat &src, cv::Size size)
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
                      255, CV_THRESH_BINARY);
        namedWindow("1", WINDOW_NORMAL);
        imshow("1", thresh);
        waitKey(0);
    }

    return 0;
}
