#include "stdafx.h"//Ԥ����ͷ�ļ�
#include <cv.h>//������Ӿ���
#include <cxcore.h>//���ݽṹ�����Դ�����
#include <highgui.h>//GUI������

int main(int argc, char *argv[]) {
//argc���������ܵĲ������� char *argv[]��һ���ַ�����
	CvHaarClassifierCascade *pCascadeFrontal = 0, *pCascadeProfile = 0;
	//��ʾHaar������������������cvLoad()�������Ӵ����м���xml�ļ���ΪHaar������������
	CvMemStorage *pStorage = cvCreateMemStorage(0);	//Ϊ�洢������洢�ռ�.���ڴ洢����
	CvSeq *pFaceRectSeq;//����һ������
	int i;
	IplImage *pInpImg = cvLoadImage("D:/������������/����ʶ��/����ʶ��test2/6.jpg",CV_LOAD_IMAGE_COLOR);
	//����cvLoadImage����ָ��ͼ���ļ���������ָ����ļ���IplImageָ��.ָ������ɫ���Խ������ͼƬתΪ3ͨ��(CV_LOAD_IMAGE_COLOR), ��ͨ�� (CV_LOAD_IMAGE_GRAYSCALE), ���߱��ֲ���(CV_LOAD_IMAGE_ANYCOLOR)
	pCascadeFrontal = (CvHaarClassifierCascade *) cvLoad ("D:/������������/����ʶ��/����ʶ��test2/haarcascade/haarcascade_frontalface_default.xml");
	pCascadeProfile = (CvHaarClassifierCascade *) cvLoad ("D:/������������/����ʶ��/����ʶ��test2/haarcascade/haarcascade_profileface.xml");

	if (!pInpImg || !pCascadeFrontal || !pCascadeProfile) {
		printf("ȱʧ�ļ�\n");
		exit(0);
	}

	cvNamedWindow("��Ƭ", CV_WINDOW_NORMAL);//����cvNamedWindow����һ�����Է���ͼ��Ĵ���   CV_WINDOW_NORMAL��������
	cvShowImage("��Ƭ", pInpImg);//�ú���Ϊ���ż�����Ӿ���OpenCV����⺯������������ָ����������ʾͼ��
	cvWaitKey(50);//cvWaitKey()�����Ĺ����ǲ���ˢ��ͼ��Ƶ��ʱ��Ϊdelay����λΪms

	pFaceRectSeq = cvHaarDetectObjects(pInpImg, pCascadeFrontal, pStorage,1.1,3,CV_HAAR_DO_CANNY_PRUNING,cvSize(40, 40));	
	//����⵽�������Ծ��ο���
	for (i=0 ; i < (pFaceRectSeq ? pFaceRectSeq->total : 0) ; i++) {
		CvRect* r = (CvRect*)cvGetSeqElem(pFaceRectSeq, i);
		CvPoint pt1 = { r->x, r->y };
		CvPoint pt2 = { r->x + r->width, r->y + r->height };
		cvRectangle(pInpImg, pt1, pt2, CV_RGB(0,255,0), 3, 4, 0);
		cvSetImageROI(pInpImg, *r);
		cvSmooth(pInpImg, pInpImg, CV_GAUSSIAN, 5, 3);
		cvResetImageROI(pInpImg);
	}
	cvShowImage("��Ƭ", pInpImg);
	cvWaitKey(1);
	//����
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

	cvShowImage("��Ƭ", pInpImg);
	cvWaitKey(0);
	cvDestroyWindow("��Ƭ");

	cvReleaseImage(&pInpImg);
	if (pCascadeFrontal) cvReleaseHaarClassifierCascade(&pCascadeFrontal);
	if (pCascadeProfile) cvReleaseHaarClassifierCascade(&pCascadeProfile);
	if (pStorage) cvReleaseMemStorage(&pStorage);
}