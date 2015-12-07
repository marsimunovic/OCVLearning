#include "sline.h"
#include <cmath>

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

bool SLine::is_parallel_to(SLine &other_line, int epsilon)
{

}

bool SLine::is_vertical_to(SLine &other_line, int epsilon)
{
    if(epsilon)
    {
        //calculate
    }
}



double SLine::inclination()
{
    return atan2(dy(), dx());
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
