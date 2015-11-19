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

TEMPLATE = app


SOURCES += main.cpp \
    key.cpp \
    calibinit.cpp
LIBS += `pkg-config opencv --libs`

HEADERS += \
    key.h \
    calibinit.h
