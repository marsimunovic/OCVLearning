#ifndef KEY_H
#define KEY_H

class Key
{
public:
    Key();

private:
    //geometrical shape of key
    enum KEY_LAYOUT
    {
        BLACK,
        WHITE_SQUARE, //full white key
        WHITE_L, //key before black doublet
        WHITE_T, //key between black doublet
        WHITE_J  //key after black doublet
    };

private:
    char tone;
    char octave_num;
    bool pressed;
    char type;
};

#endif // KEY_H
