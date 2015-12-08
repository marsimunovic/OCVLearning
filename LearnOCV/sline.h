#ifndef SLINE_H
#define SLINE_H


void testLine();

class SLine
{
public:
    struct Point
    {
        Point():x(0), y(0){}
        Point(int x_, int y_) : x(x_), y(y_){}
        int x, y;
    };

public:
    SLine();
    SLine(int x1, int y1, int x2, int y2);
    Point start_point(){return start;}
    Point end_point(){return end;}

    static bool compare_by_x(SLine const& first, SLine const &second);
    static bool compare_by_y(SLine const& first, SLine const &second);

    bool is_parallel_to(SLine& other_line, int epsilon = 0);
    bool is_vertical_to(SLine& other_line, int epsilon = 0);
    double inclination();
    inline double norm();
    int dot_product(SLine& other_line);
    void to_upwardy();
    void to_righty();

    double inclination_4quadrant();
private:
    Point start, end;
    inline int dx();
    inline int dy();
};

#endif // SLINE_H
