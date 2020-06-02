#ifndef __MINST_
#define __MINST_
/*
MINST数据库是一个手写图像数据库，里面
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>


//单张图像的结构体
typedef struct MinstImg{
	int w;           // 图像宽w
	int h;           // 图像高h
	int ch;          // 图像的通道数
	float* ImgData;  // 1维float类型图像数据指针
}MinstImg;
// 读入图像到MinstImg结构体
MinstImg read_Img(const char* filename);      
/*
读取权重，返回float类型的指针
*/
float* read_weight(const char* filename, int fsize, int inchan, int outchan);
/*
读取偏置，返回float类型的指针
*/
float* read_bias(const char* filename,int num_filter);

////保存ImgNum张图片结构体的结构体
//typedef struct MinstImgArr{
//	int ImgNum;        // 存储图像的数量
//	MinstImg* ImgPtr;  // 存储图像数组指针
//}*ImgArr;              // 存储图像数据的数组


//标签结构体
typedef struct MinstLabel{
	int l;            // 输出便签的维度
	float* LabelData; // 输出标记数据
}MinstLabel;
//保存标签的结构体
typedef struct MinstLabelArr{
	int LabelNum;
	MinstLabel* LabelPtr;
}*LabelArr;              // 存储图像标记的数组

//void save_Img(ImgArr imgarr,char* filedir); // 将图像数据提取到结构体中
//
//LabelArr read_Lable(const char* filename); // 读入图像标记




char * combine_strings(char *a, char *b);
#endif