#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include<iostream>
#include "noise.h"
#include "filter.h"

using namespace std;
using namespace cv;


int main() {
	//加载灰度图像或彩色图像
	Mat img1 = imread("C:\\Users\\Lenovo\\Desktop\\1.jpg", 1);
	imshow("原灰度图像", img1);
	//waitKey(0);

	//添加高斯噪声
	//Mat gaussianImg = addGaussianNoise(img1);
	//imshow("添加高斯噪声", gaussianImg);
	//waitKey(0);

	//添加胡椒噪声
	//Mat pepImg = addPepNoise(img1, 3000);
	//imshow("添加椒盐噪声", pepImg);
	//waitKey(0);
	
	//添加盐噪声
	//Mat saltImg = addSaltNoise(img1, 3000);
	//imshow("添加盐噪声", saltImg);
	//waitKey(0);

	//添加椒盐噪声
	Mat pepSaltImg = addPepSaltNoise(img1, 3000);
	imshow("添加椒盐噪声", pepSaltImg);
	//waitKey(0);

	//1.均值滤波
	aveFilter(pepSaltImg);

	//2.几何均值滤波
	geoAveFilter(pepSaltImg);

	//3.谐波均值滤波
	//harAveFilter(pepSaltImg);

	//4.逆谐波滤波器
	//conAveFilter(pepSaltImg, 1);

	//5.中值滤波
	//medianFilter(pepSaltImg, 5);
	//medianFilter(pepSaltImg, 9);

	//6.自适应均值滤波
	//adaptMeanFilter(pepSaltImg);

	//7.自适应中值滤波
	//adaptMedianFilter(pepSaltImg);

}