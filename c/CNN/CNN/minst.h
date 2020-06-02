#ifndef __MINST_
#define __MINST_
/*
MINST���ݿ���һ����дͼ�����ݿ⣬����
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>


//����ͼ��Ľṹ��
typedef struct MinstImg{
	int w;           // ͼ���w
	int h;           // ͼ���h
	int ch;          // ͼ���ͨ����
	float* ImgData;  // 1άfloat����ͼ������ָ��
}MinstImg;
// ����ͼ��MinstImg�ṹ��
MinstImg read_Img(const char* filename);      
/*
��ȡȨ�أ�����float���͵�ָ��
*/
float* read_weight(const char* filename, int fsize, int inchan, int outchan);
/*
��ȡƫ�ã�����float���͵�ָ��
*/
float* read_bias(const char* filename,int num_filter);

////����ImgNum��ͼƬ�ṹ��Ľṹ��
//typedef struct MinstImgArr{
//	int ImgNum;        // �洢ͼ�������
//	MinstImg* ImgPtr;  // �洢ͼ������ָ��
//}*ImgArr;              // �洢ͼ�����ݵ�����


//��ǩ�ṹ��
typedef struct MinstLabel{
	int l;            // �����ǩ��ά��
	float* LabelData; // ����������
}MinstLabel;
//�����ǩ�Ľṹ��
typedef struct MinstLabelArr{
	int LabelNum;
	MinstLabel* LabelPtr;
}*LabelArr;              // �洢ͼ���ǵ�����

//void save_Img(ImgArr imgarr,char* filedir); // ��ͼ��������ȡ���ṹ����
//
//LabelArr read_Lable(const char* filename); // ����ͼ����




char * combine_strings(char *a, char *b);
#endif