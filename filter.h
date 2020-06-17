#pragma once
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include<iostream>

using namespace std;
using namespace cv;

//������ֵ�˲�
void aveFilter(Mat src);

//���ξ�ֵ�˲�
void geoAveFilter(Mat src);

//г����ֵ�˲�
void harAveFilter(Mat src);

//��г����ֵ�˲�
void conAveFilter(Mat src, int Q);

//��ֵ�˲�
void medianFilter(Mat src, int size);

//����Ӧ��ֵ�˲�
uchar adaptMean(Mat src, int row, int col);
void adaptMeanFilter(Mat src);


//����Ӧ��ֵ�˲�
uchar adaptMedian(Mat src, int row, int col, int size, int maxsize);
void adaptMedianFilter(Mat src);


