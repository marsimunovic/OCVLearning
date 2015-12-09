#include "sline.h"
#include <cmath>
#include <iostream>


inline double RAD_TO_DEGREE(double rad)
{
    return 180.0*rad/M_PI;
}

inline double DEGREE_TO_RAD(double degree)
{
    return degree*M_PI/180.0;
}

SLine::SLine()
    :start(0,0)
    ,end(0,0)
{
}

SLine::SLine(int x1, int y1, int x2, int y2)
    :start(x1, y1)
    ,end(x2, y2)
{
}

bool SLine::compare_by_x(SLine const &first, SLine const &second)
{
    return (first.start.x + first.end.x) < (second.start.x + second.end.x);
}

bool SLine::compare_by_y(SLine const &first, SLine const &second)
{
    return (first.start.y + first.end.y) < (second.start.y + second.end.y);
}

bool SLine::is_parallel_to(SLine &other_line, int epsilon)
{
    double dot_prod = static_cast<double>(dot_product(other_line));
    if(dot_prod < 0.0000001)
        return false;
    double norm1 = norm();
    double norm2 = other_line.norm();
    double cosfi = dot_prod/norm1/norm2;
    double fi = acos(cosfi);
    double diff = std::abs(DEGREE_TO_RAD(0.0)-fi);
    if(epsilon)
    {
        if(diff < epsilon)
        //calculate
            return true;
    }
    else
    {
        return static_cast<bool>(std::abs(fi) < 0.000001);
    }
    return false;
}

bool SLine::is_vertical_to(SLine &other_line, int epsilon)
{
    int dot_prod = dot_product(other_line);
    if(epsilon)
    {
        //calculate
        double norm1 = norm();
        double norm2 = other_line.norm();
        double cosfi = dot_prod/norm1/norm2;
        double fi = acos(cosfi);
        if(std::abs(DEGREE_TO_RAD(90.0)-fi) < epsilon)
            return true;
    }
    else
        return static_cast<bool>(std::abs(dot_prod) < 0.000001);
    return false;
}

void SLine::to_upwardy()
{
    if(start.y < end.y)
        std::swap(start, end);
}

void SLine::to_righty()
{
    if(start.x > end.x)
        std::swap(start, end);
}

double SLine::inclination()
{
    return abs(atan(abs(dy()/(dx() + 0.00000000001))));
}

double SLine::inclination_4quadrant()
{
    return atan2(dy(),(dx() + 0.00000000001));
}

int SLine::dot_product(SLine &other_line)
{
    return dx()*other_line.dx()+ dy()*other_line.dy();
}


int SLine::dx()
{
    return end.x - start.x;
}

int SLine::dy()
{
    return end.y - start.y;
}

double SLine::norm()
{
    return sqrt(dx()*dx() + dy()*dy());
}

double SLine::calculate_angle( SLine::Point pt1, SLine::Point pt2, SLine::Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


void testLine()
{
    SLine l1(0, 0, 3, 0);
    SLine l2(0, 3, 3, 3);
    std::cout << "|l1| = " << l1.norm() << " : |l2| = " << l2.norm() << std::endl;
    std::cout << "l1 dot l2 = " << l1.dot_product(l2) << std::endl;
    std::cout << "l1 is" << (l1.is_vertical_to(l2)? "" : " not") << " vertical to l2" << std::endl;
    std::cout << "l1 is" << (l1.is_parallel_to(l2)? "" : " not") << " parallel to l2" << std::endl;
}
