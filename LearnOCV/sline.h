#ifndef SLINE_H
#define SLINE_H

class SLine
{
private:
    struct Point
    {
        Point():x(0), y(0){}
        Point(int x_, int y_) : x(x_), y(y_){}
        int x, y;
    };

public:
    SLine();
    SLine(int x1, int y1, int x2, int y2);

    bool is_parallel_to(SLine& other_line, int epsilon = 0);
    bool is_vertical_to(SLine& other_line, int epsilon = 0);
    double inclination();
private:
    Point start, end;
    int dot_product(SLine& other_line);
    inline int dx();
    inline int dy();
    inline double norm();
};

#endif // SLINE_H
