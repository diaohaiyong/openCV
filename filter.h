#pragma once
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include<iostream>

using namespace std;
using namespace cv;

//算数均值滤波
void aveFilter(Mat src);

//几何均值滤波
void geoAveFilter(Mat src);

//谐波均值滤波
void harAveFilter(Mat src);

//逆谐波均值滤波
void conAveFilter(Mat src, int Q);

//中值滤波
void medianFilter(Mat src, int size);

//自适应均值滤波
uchar adaptMean(Mat src, int row, int col);
void adaptMeanFilter(Mat src);


//自适应中值滤波
uchar adaptMedian(Mat src, int row, int col, int size, int maxsize);
void adaptMedianFilter(Mat src);


