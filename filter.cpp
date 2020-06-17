#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include<iostream>
#include "filter.h" 

using namespace std;
using namespace cv;

//算数均值滤波
void aveFilter(Mat src) {
	Mat res = src.clone();
	int rows = res.rows;
	int cols = res.cols;
	int channel = res.channels();
	for (int i = 2; i < rows - 2; i++) {
		for (int j = 2; j < cols - 2; j++) {
			if (channel == 1) {
				int sum = 0;
				for (int m = i - 2; m <= i + 2; m++) {
					for (int n = j - 2; n <= j + 2; n++) {
						sum += src.at<uchar>(m, n);
					}
				}
				res.at<uchar>(i, j) = sum / 25;
			}
			else {
				int b = 0, g = 0, r = 0;
				for (int m = i - 2; m <= i + 2; m++) {
					for (int n = j - 2; n <= j + 2; n++) {
						r += src.at<Vec3b>(m, n)[0];
						g += src.at<Vec3b>(m, n)[1];
						b += src.at<Vec3b>(m, n)[2];
					}
				}
				res.at<Vec3b>(i, j)[0] = r / 25;
				res.at<Vec3b>(i, j)[1] = g / 25;
				res.at<Vec3b>(i, j)[2] = b / 25;
			}
		}
	}
	imshow("均值滤波", res);
	waitKey(0);
	return;
}

//几何均值滤波器
void geoAveFilter(Mat src) {
	Mat res = src.clone();
	int rows = res.rows;
	int cols = res.cols;
	int channel = res.channels();
	for (int i = 2; i < rows - 2; i++) {
		for (int j = 2; j < cols - 2; j++) {
			if (channel == 1) {
				double sum = 1;
				for (int m = i - 2; m <= i + 2; m++) {
					for (int n = j - 2; n <= j + 2; n++) {
						if (src.at<uchar>(m, n) != 0) {
							sum *= src.at<uchar>(m, n);
						}
					}
				}
				int ptr = pow(sum, 1.0 / double(25));
				if (ptr < 0)
					ptr = 0;
				if (ptr > 255)
					ptr = 255;
				res.at<uchar>(i, j) = static_cast<uchar>(ptr);
			}
			else {
				double r = 1, g = 1, b = 1;
				for (int m = i - 2; m <= i + 2; m++) {
					for (int n = j - 2; n <= j + 2; n++) {
						if (src.at<Vec3b>(m, n)[0] != 0)
							r *= src.at<Vec3b>(m, n)[0];
						if (src.at<Vec3b>(m, n)[1] != 0)
							g *= src.at<Vec3b>(m, n)[1];
						if (src.at<Vec3b>(m, n)[2] != 0)
							b *= src.at<Vec3b>(m, n)[2];
					}
				}
				res.at<Vec3b>(i, j)[0] = pow(r, 1.0 / double(25));
				res.at<Vec3b>(i, j)[1] = pow(g, 1.0 / double(25));
				res.at<Vec3b>(i, j)[2] = pow(b, 1.0 / double(25));
			}
		}
	}
	imshow("几何均值滤波", res);
	waitKey(0);
	return;
}

//谐波均值滤波器
void harAveFilter(Mat src) {
	Mat res = src.clone();

	int rows = src.rows;
	int cols = src.cols;
	for (int i = 2; i < rows - 2; i++) {
		for (int j = 2; j < cols - 2; j++) {
			double sum = 0;
			for (int m = i - 2; m <= i + 2; m++) {
				for (int n = j - 2; n <= j + 2; n++) {
					if (src.at<uchar>(m, n) != 0) {
						sum += 1.0 / double(src.at<uchar>(m, n));
					}
				}
			}
			res.at<uchar>(i, j) = int(25.0 / sum);
		}
	}
	imshow("谐波滤波器", res);
	waitKey(0);
}

//逆谐波滤波器
void conAveFilter(Mat src, int Q) {
	Mat res = src.clone();
	int rows = src.rows;
	int cols = src.cols;
	for (int i = 2; i < rows - 2; i++) {
		for (int j = 2; j < cols - 2; j++) {
			int sum1 = 0;
			int sum2 = 0;
			for (int m = i - 2; m <= i + 2; m++) {
				for (int n = j - 2; n <= j + 2; n++) {
					sum1 += pow(src.at<uchar>(m, n), Q + 1);
					sum2 += pow(src.at<uchar>(m, n), Q);
				}
			}
			res.at<uchar>(i, j) = sum1 / sum2;
		}
	}
	imshow("逆谐波滤波", res);
	waitKey(0);
	return;
}

//实现中值滤波
void medianFilter(Mat src, int size) {
	Mat res = src.clone();
	int rows = src.rows;
	int cols = src.cols;
	for (int i = size / 2; i < rows - size / 2; i++) {
		for (int j = size / 2; j < cols - size / 2; j++) {
			vector<int> median;
			for (int m = i - size / 2; m <= i + size / 2; m++) {
				for (int n = j - size / 2; n <= j + size / 2; n++) {
					median.push_back(src.at<uchar>(m, n));
				}
			}
			sort(median.begin(), median.end());
			res.at<uchar>(i, j) = median[size*size / 2];
		}
	}
	imshow("中值滤波", res);
	waitKey(0);
	return;
}


//自适应均值滤波
//产生自适应像素
uchar adaptMean(Mat src, int row, int col) {
	int size = 7;
	int sigma_n = 3000;
	vector<uchar> pixes;
	for (int a = -size / 2; a <= size / 2; a++) {
		for (int b = -size / 2; b <= size / 2; b++) {
			pixes.push_back(src.at<uchar>(row + a, col + b));
		}
	}
	//得到均值
	int sum = 0;
	for (int i = 0; i < pixes.size(); i++) {
		sum += pixes[i];
	}
	int mean = sum / pixes.size();

	//得到方差
	sum = 0;
	for (int i = 0; i < pixes.size(); i++) {
		sum += pow((pixes[i] - mean), 2);
	}
	int sigma = sum / pixes.size();

	double rate = double(sigma_n) / double(sigma);
	if (rate > 1.0)
		rate = 1.0;
	int value = src.at<uchar>(row, col);
	return value - rate*(value - mean);
}
void adaptMeanFilter(Mat src) {
	Mat res = src.clone();
	int rows = src.rows;
	int cols = src.cols;
	for (int i = 3; i < rows - 3; i++) {
		for (int j = 3; j < cols - 3; j++) {
			res.at<uchar>(i, j) = adaptMean(src, i, j);
		}
	}
	imshow("自适应均值滤波", res);
	waitKey(0);
	return;
}


//自适应中值滤波
//产生自适应像素
uchar adaptMedian(Mat im, int row, int col, int kernelSize, int maxSize) {
	vector<uchar> pixels;
	for (int a = -kernelSize / 2; a <= kernelSize / 2; a++)
		for (int b = -kernelSize / 2; b <= kernelSize / 2; b++)
		{
			pixels.push_back(im.at<uchar>(row + a, col + b));
		}
	sort(pixels.begin(), pixels.end());
	auto min = pixels[0];
	auto max = pixels[kernelSize * kernelSize - 1];
	auto med = pixels[kernelSize * kernelSize / 2];
	auto zxy = im.at<uchar>(row, col);
	if (med > min && med < max)
	{
		// to B
		if (zxy > min && zxy < max)
			return zxy;
		else
			return med;
	}
	else
	{
		kernelSize += 2;
		if (kernelSize <= maxSize)
			return adaptMedian(im, row, col, kernelSize, maxSize); // 增大窗口尺寸，继续A过程。
		else
			return med;
	}
}
//自适应中值滤波
void adaptMedianFilter(Mat im) {
	int minSize = 3; // 滤波器窗口的起始尺寸
	int maxSize = 7; // 滤波器窗口的最大尺寸
	Mat res;
	// 扩展图像的边界
	copyMakeBorder(im, res, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, BorderTypes::BORDER_REFLECT);
	// 图像循环
	for (int j = maxSize / 2; j < res.rows - maxSize / 2; j++)
	{
		for (int i = maxSize / 2; i < res.cols * res.channels() - maxSize / 2; i++)
		{
			res.at<uchar>(j, i) = adaptMedian(res, j, i, minSize, maxSize);
		}
	}
	imshow("自适应中值滤波", res);
	waitKey(0);
	return;
}