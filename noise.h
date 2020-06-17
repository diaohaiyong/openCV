#pragma once
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include<iostream>

using namespace std;
using namespace cv;

//产生高斯噪声
double generateGaussianNoise(double mu, double sigma);
Mat addGaussianNoise(Mat src);

//产生胡椒噪声
Mat addPepNoise(Mat src, int n);

//产生盐噪声
Mat addSaltNoise(Mat src, int n);

//产生椒盐噪声
Mat addPepSaltNoise(Mat src, int n);


