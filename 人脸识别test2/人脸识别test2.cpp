#include "stdafx.h"//预编译头文件
#include <cv.h>//计算机视觉库
#include <cxcore.h>//数据结构与线性代数库
#include <highgui.h>//GUI函数库

int main(int argc, char *argv[]) {
//argc是命令行总的参数个数 char *argv[]是一个字符数组
	CvHaarClassifierCascade *pCascadeFrontal = 0, *pCascadeProfile = 0;
	//表示Haar特征分类器，可以用cvLoad()函数来从磁盘中加载xml文件作为Haar特征分类器。
	CvMemStorage *pStorage = cvCreateMemStorage(0);	//为存储器申请存储空间.用于存储序列
	CvSeq *pFaceRectSeq;//创建一个序列
	int i;
	IplImage *pInpImg = cvLoadImage("D:/软件工程与计算/人脸识别/人脸识别test2/6.jpg",CV_LOAD_IMAGE_COLOR);
	//函数cvLoadImage载入指定图像文件，并返回指向该文件的IplImage指针.指定的颜色可以将输入的图片转为3通道(CV_LOAD_IMAGE_COLOR), 单通道 (CV_LOAD_IMAGE_GRAYSCALE), 或者保持不变(CV_LOAD_IMAGE_ANYCOLOR)
	pCascadeFrontal = (CvHaarClassifierCascade *) cvLoad ("D:/软件工程与计算/人脸识别/人脸识别test2/haarcascade/haarcascade_frontalface_default.xml");
	pCascadeProfile = (CvHaarClassifierCascade *) cvLoad ("D:/软件工程与计算/人脸识别/人脸识别test2/haarcascade/haarcascade_profileface.xml");

	if (!pInpImg || !pCascadeFrontal || !pCascadeProfile) {
		printf("缺失文件\n");
		exit(0);
	}

	cvNamedWindow("照片", CV_WINDOW_NORMAL);//函数cvNamedWindow创建一个可以放置图像的窗口   CV_WINDOW_NORMAL窗口属性
	cvShowImage("照片", pInpImg);//该函数为开放计算机视觉（OpenCV）库库函数，用来在在指定窗口中显示图像。
	cvWaitKey(50);//cvWaitKey()函数的功能是不断刷新图像，频率时间为delay，单位为ms

	pFaceRectSeq = cvHaarDetectObjects(pInpImg, pCascadeFrontal, pStorage,1.1,3,CV_HAAR_DO_CANNY_PRUNING,cvSize(40, 40));	
	//将检测到的人脸以矩形框标出
	for (i=0 ; i < (pFaceRectSeq ? pFaceRectSeq->total : 0) ; i++) {
		CvRect* r = (CvRect*)cvGetSeqElem(pFaceRectSeq, i);
		CvPoint pt1 = { r->x, r->y };
		CvPoint pt2 = { r->x + r->width, r->y + r->height };
		cvRectangle(pInpImg, pt1, pt2, CV_RGB(0,255,0), 3, 4, 0);
		cvSetImageROI(pInpImg, *r);
		cvSmooth(pInpImg, pInpImg, CV_GAUSSIAN, 5, 3);
		cvResetImageROI(pInpImg);
	}
	cvShowImage("照片", pInpImg);
	cvWaitKey(1);
	//侧脸
	pFaceRectSeq = cvHaarDetectObjects
		(pInpImg, pCascadeProfile, pStorage,1.4,3,CV_HAAR_DO_CANNY_PRUNING,cvSize(0, 0));

	for (i=0 ; i < (pFaceRectSeq ? pFaceRectSeq->total : 0) ; i++) {
		CvRect* r = (CvRect*)cvGetSeqElem(pFaceRectSeq, i);
		CvPoint pt1 = { r->x, r->y };
		CvPoint pt2 = { r->x + r->width, r->y + r->height };
		cvRectangle(pInpImg, pt1, pt2, CV_RGB(255,165,0), 3, 4, 0);
		cvSetImageROI(pInpImg, *r);
		cvSmooth(pInpImg, pInpImg, CV_GAUSSIAN, 5, 3);
		cvResetImageROI(pInpImg);
	}

	cvShowImage("照片", pInpImg);
	cvWaitKey(0);
	cvDestroyWindow("照片");

	cvReleaseImage(&pInpImg);
	if (pCascadeFrontal) cvReleaseHaarClassifierCascade(&pCascadeFrontal);
	if (pCascadeProfile) cvReleaseHaarClassifierCascade(&pCascadeProfile);
	if (pStorage) cvReleaseMemStorage(&pStorage);
}