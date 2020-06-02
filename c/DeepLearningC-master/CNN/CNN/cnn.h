#ifndef __CNN_
#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>

#include "minst.h"

#define AvePool 0          //不同的池化类型对应的枚举值
#define MaxPool 1 
#define MinPool 2

//卷积运算(Relu激活进行卷积计算，softmax激活是进行全连接计算）
//参数：该层的权重指针，偏置指针，输入图像数据指针、输入宽、输入高、输入通道数、输出通道数、卷积核的大小、步幅大小、补边否（1是进行补边,0是不补边）,激活函数（1是relu,0是softmax）
float* conv(float* weigth, float* bias,float* imgData,int inw,int inh,int inchan,int outchan,int fsize,int stride,int pad,int activation); // 卷积操作
//获取pad之前的特征图上一点的像素值。参数：padding前的图像数据指针、高、宽，padding后的通道数、行、列，pad大小
float get_pixel(float *im, int inh,int inw,  int channel,int row, int col,int pad);
////im2col操作
//float**  im2col_cpu(float* data_im,int channels, int height, int width,int ksize, int stride, int pad, float* data_col);
//float im2col_get_pixel(float *im, int height, int width, int channels, int row, int col, int channel, int pad);
////gemm矩阵乘操作
//float** gemm(int TA, int TB, int M, int N, int K, float ALPHA,float *A, int lda,float *B, int ldb,float BETA,float *C, int ldc);
//

//最大池化运算
//参数：            输入图像、输入高、输入宽、输入通道数、池化核大小，步幅大小
float* maxpool(float* imgData, int inh, int inw, int inchan,    int psize, int stride);

//全连接运算
//参数： 该层的权重指针，偏置指针，输入图像数据指针，权重行数，权重列数，激活类型
float* nn(float* weigth, float* bias, float* imgData, int weigthrow, int weigthcol, int activation); // NN操作
//输出全连接层计算(激活函数也是soft)
//
char* intTochar(int i);

//// 卷积层
//typedef struct convolutional_layer{
//	int inputWidth;   //输入图像的宽
//	int inputHeight;  //输入图像的长
//	int mapSize;      //特征模板的大小，模板一般都是正方形
//	int inChannels;   //输入图像的通道数
//	int outChannels;  //输出图像的通道数
//	int pad;          //padding大小
//	int stride;       //步幅大小
//	// 关于特征模板的权重分布，这里是一个四维数组
//	// 其大小为inChannels*outChannels*mapSize*mapSize大小
//	// 这里用四维数组，主要是为了表现全连接的形式，实际上卷积层并没有用到全连接的形式
//	// 这里的例子是DeapLearningToolboox里的CNN例子，其用到就是全连接
//	float**** mapData;     //存放四维特征权重的数据
//
//	float* basicData;   //偏置，偏置的大小，为outChannels
//	bool isFullConnect; //是否为全连接
//	bool* connectModel; //连接模式（默认为全连接）  
//
//	// 下面三者的大小同输出的维度相同
//	float*** v;			// 激活函数前的三维输入值
//	float*** y;			// 激活函数后神经元的三维输出
//
//}CovLayer;
//
//// 采样层 pooling
//typedef struct pooling_layer{
//	int inputWidth;   //输入图像的宽
//	int inputHeight;  //输入图像的长
//	int mapSize;      //特征模板的大小
//	int inChannels;   //输入图像的数目
//	int outChannels;  //输出图像的数目
//
//	int poolType;     //Pooling的方法
//	float* basicData;   //偏置
//
//	float*** y; // 采样函数后神经元的三维输出,无激活函数
//	//float*** d; // 网络的局部梯度,δ值
//}PoolLayer;
//
//// 输出层 全连接的神经网络
//typedef struct nn_layer{
//	int inputNum;   //输入数据的数目
//	int outputNum;  //输出数据的数目
//	float** wData; // 两维权重数据，为一个inputNum*outputNum大小
//	float* basicData;   //偏置，大小为outputNum大小
//
//	// 下面三者的大小与输出的维度相同
//	float* v; // 进入激活函数的输入值
//	float* y; // 激活函数后神经元的输出
//	//float* d; // 网络的局部梯度,δ值
//
//	bool isFullConnect; //是否为全连接
//}OutLayer;
//
////网络结构
//typedef struct cnn_network{
//	int layerNum;    //网络的总层数
//	CovLayer* C1;     //卷积层结构体变量C1
//	PoolLayer* S2;          //池化层结构体变量S2
//	CovLayer* C3;     //卷积层结构体变量C3
//	PoolLayer* S4;         //池化层结构体变量S4
//	OutLayer* O5;    //全连接层的结构体变量05
//	OutLayer* O6;    //输出层结构体变量06
//
//	//float* e; // 训练误差
//	float* L; // 瞬时误差能量
//}CNN;
//
////训练参数
//typedef struct train_opts{
//	int numepochs; // 训练的迭代次数
//	float alpha; // 学习速率
//}CNNOpts;


// cnn初始化网络
//void cnnsetup(CNN* cnn,nSize inputSize,int outputSize);
/*	
	CNN网络的训练函数
	inputData，outputData分别存入训练数据
	trainNum表明数据数目
*/
//void cnntrain(CNN* cnn,	ImgArr inputData,LabelArr outputData,CNNOpts opts,int trainNum);
//
//// 测试cnn函数
//float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum);
//
//// 保存cnn
//void savecnn(CNN* cnn, const char* filename);
//
//// 导入cnn的数据
//void importcnn(CNN* cnn,  char* filename);

//// 初始化卷积层，  7参数： 输入宽、输入高、卷积核尺寸、输入通道数、输出通道数，padding大小，步幅大小
//CovLayer* initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels,int pad, int stride);
//void CovLayerConnect(CovLayer* covL,bool* connectModel);
//
//// 初始化采样层
//PoolLayer* initPoolLayer(int inputWidth,int inputHeigh,int mapSize,int inChannels,int outChannels,int poolType);
//void PoolLayerConnect(PoolLayer* poolL,bool* connectModel);
//
//// 初始化输出层
//OutLayer* initOutLayer(int inputNum,int outputNum);

// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
float activation_Sigma(float input,float bas); // sigma激活函数

//void cnnff(CNN* cnn,float** inputData); // 网络的前向传播
//void cnnbp(CNN* cnn,float* outputData); // 网络的后向传播
//void cnnapplygrads(CNN* cnn,CNNOpts opts,float** inputData);
//void cnnclear(CNN* cnn); // 将数据vyd清零

//合并文件mergeFile (infile1, infile2, filenmae)
void mergeFile(const char *fp1, const char *fp2);

/*
	Pooling Function
	input 输入数据
	inputNum 输入数据数目
	mapSize 求平均的模块区域
*/
//void avgPooling(float** output,nSize outputSize,float** input,nSize inputSize,int mapSize); // 求平均值
/* 
	单层全连接神经网络的处理
	nnSize是网络的大小
*/
//void nnff(float* output,float* input,float** wdata,float* bas,nSize nnSize); // 单层全连接神经网络的前向传播

//void savecnndata(CNN* cnn,const char* filename,float** inputdata); // 保存CNN网络中的相关数据

#endif
