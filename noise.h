#pragma once
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include<iostream>

using namespace std;
using namespace cv;

//������˹����
double generateGaussianNoise(double mu, double sigma);
Mat addGaussianNoise(Mat src);

//������������
Mat addPepNoise(Mat src, int n);

//����������
Mat addSaltNoise(Mat src, int n);

//������������
Mat addPepSaltNoise(Mat src, int n);


