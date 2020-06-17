#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include<iostream>
#include "noise.h"

using namespace std;
using namespace cv;

//高斯噪声
double generateGaussianNoise(double mu, double sigma)
{
	//定义小值
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flag为假构造高斯随机变量X
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//构造随机变量
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flag为真构造高斯随机变量
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0*sigma + mu;
}

Mat addGaussianNoise(Mat src) {
	Mat res = src.clone();
	int rows = src.rows;
	int cols = src.cols;
	int channel = src.channels();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (channel == 1) {
				int val = src.at<uchar>(i, j) + generateGaussianNoise(2, 0.8) * 32;
				val = val < 0 ? 0 : val;
				val = val > 255 ? 255 : val;
				res.at<uchar>(i, j) = val;
			}
			else {
				int r = src.at<Vec3b>(i, j)[0] + generateGaussianNoise(2, 0.8) * 32;
				int g = src.at<Vec3b>(i, j)[1] + generateGaussianNoise(2, 0.8) * 32;
				int b = src.at<Vec3b>(i, j)[2] + generateGaussianNoise(2, 0.8) * 32;
				r = r < 0 ? 0 : r;
				r = r > 255 ? 255 : r;
				g = g < 0 ? 0 : g;
				g = g > 255 ? 255 : g;
				b = b < 0 ? 0 : b;
				b = b > 255 ? 255 : b;
				res.at<Vec3b>(i, j)[0] = r;
				res.at<Vec3b>(i, j)[1] = g;
				res.at<Vec3b>(i, j)[2] = b;
			}
		}
	}
	return res;
}

//胡椒噪声
Mat addPepNoise(Mat src, int n) {
	Mat res = src.clone();
	int rows = src.rows;
	int cols = src.cols;
	int channel = src.channels();
	for (int i = 0; i < n; i++) {
		int row = rand() % rows;
		int col = rand() % cols;
		if (channel == 1) {
			res.at<uchar>(row, col) = 0;
		}
		else {
			res.at<Vec3b>(row, col)[0] = 0;
			res.at<Vec3b>(row, col)[1] = 0;
			res.at<Vec3b>(row, col)[2] = 0;
		}
	}
	return res;
}

//盐噪声，随机将像素置为255
Mat addSaltNoise(Mat src, int n) {
	Mat res = src.clone();
	int rows = src.rows;
	int cols = src.cols;
	int channel = src.channels();
	for (int i = 0; i < n; i++) {
		int row = rand() % rows;
		int col = rand() % cols;
		if (channel == 1) {
			res.at<uchar>(row, col) = 255;
		}
		else {
			res.at<Vec3b>(row, col)[0] = 255;
			res.at<Vec3b>(row, col)[1] = 255;
			res.at<Vec3b>(row, col)[2] = 255;
		}
	}
	return res;
}

//产生椒盐噪声
Mat addPepSaltNoise(Mat src, int n) {
	Mat res = src.clone();
	int rows = src.rows;
	int cols = src.cols;
	int channel = src.channels();
	//置255
	for (int i = 0; i < n; i++) {
		int row = rand() % rows;
		int col = rand() % cols;
		if (channel == 1) {
			res.at<uchar>(row, col) = 255;
		}
		else {
			res.at<Vec3b>(row, col) = 255;
			res.at<Vec3b>(row, col) = 255;
			res.at<Vec3b>(row, col) = 255;
		}
	}
	//置0
	for (int i = 0; i < n; i++) {
		int row = rand() % rows;
		int col = rand() % cols;
		if (channel == 1) {
			res.at<uchar>(row, col) = 0;
		}
		else {
			res.at<Vec3b>(row, col)[0] = 0;
			res.at<Vec3b>(row, col)[1] = 0;
			res.at<Vec3b>(row, col)[2] = 0;
		}
	}
	return res;
}