#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include<iostream>
#include "noise.h"
#include "filter.h"

using namespace std;
using namespace cv;


int main() {
	//���ػҶ�ͼ����ɫͼ��
	Mat img1 = imread("C:\\Users\\Lenovo\\Desktop\\1.jpg", 1);
	imshow("ԭ�Ҷ�ͼ��", img1);
	//waitKey(0);

	//��Ӹ�˹����
	//Mat gaussianImg = addGaussianNoise(img1);
	//imshow("��Ӹ�˹����", gaussianImg);
	//waitKey(0);

	//��Ӻ�������
	//Mat pepImg = addPepNoise(img1, 3000);
	//imshow("��ӽ�������", pepImg);
	//waitKey(0);
	
	//���������
	//Mat saltImg = addSaltNoise(img1, 3000);
	//imshow("���������", saltImg);
	//waitKey(0);

	//��ӽ�������
	Mat pepSaltImg = addPepSaltNoise(img1, 3000);
	imshow("��ӽ�������", pepSaltImg);
	//waitKey(0);

	//1.��ֵ�˲�
	aveFilter(pepSaltImg);

	//2.���ξ�ֵ�˲�
	geoAveFilter(pepSaltImg);

	//3.г����ֵ�˲�
	//harAveFilter(pepSaltImg);

	//4.��г���˲���
	//conAveFilter(pepSaltImg, 1);

	//5.��ֵ�˲�
	//medianFilter(pepSaltImg, 5);
	//medianFilter(pepSaltImg, 9);

	//6.����Ӧ��ֵ�˲�
	//adaptMeanFilter(pepSaltImg);

	//7.����Ӧ��ֵ�˲�
	//adaptMedianFilter(pepSaltImg);

}