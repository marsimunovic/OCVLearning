#-------------------------------------------------
#
# Project created by QtCreator 2015-11-18T10:18:21
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = LearnOCV
CONFIG   += console
CONFIG   -= app_bundle
QMAKE_CXXFLAGS += -std=c++11

TEMPLATE = app


SOURCES += \
    key.cpp \
    calibinit.cpp \
    detectkeyboard.cpp \
    main1.cpp \
    main.cpp \
    detectkeys.cpp \
    sline.cpp \
    math_functions.cpp \
    cvline.cpp
LIBS += `pkg-config opencv --libs`

HEADERS += \
    key.h \
    calibinit.h \
    detectkeyboard.h \
    detectkeys.h \
    sline.h \
    math_functions.h \
    cvline.h
