#ifndef __CNN_
#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>

#include "minst.h"

#define AvePool 0          //��ͬ�ĳػ����Ͷ�Ӧ��ö��ֵ
#define MaxPool 1 
#define MinPool 2

//�������(Relu������о�����㣬softmax�����ǽ���ȫ���Ӽ��㣩
//�������ò��Ȩ��ָ�룬ƫ��ָ�룬����ͼ������ָ�롢���������ߡ�����ͨ���������ͨ����������˵Ĵ�С��������С�����߷�1�ǽ��в���,0�ǲ����ߣ�,�������1��relu,0��softmax��
float* conv(float* weigth, float* bias,float* imgData,int inw,int inh,int inchan,int outchan,int fsize,int stride,int pad,int activation); // �������
//��ȡpad֮ǰ������ͼ��һ�������ֵ��������paddingǰ��ͼ������ָ�롢�ߡ���padding���ͨ�������С��У�pad��С
float get_pixel(float *im, int inh,int inw,  int channel,int row, int col,int pad);
////im2col����
//float**  im2col_cpu(float* data_im,int channels, int height, int width,int ksize, int stride, int pad, float* data_col);
//float im2col_get_pixel(float *im, int height, int width, int channels, int row, int col, int channel, int pad);
////gemm����˲���
//float** gemm(int TA, int TB, int M, int N, int K, float ALPHA,float *A, int lda,float *B, int ldb,float BETA,float *C, int ldc);
//

//���ػ�����
//������            ����ͼ������ߡ����������ͨ�������ػ��˴�С��������С
float* maxpool(float* imgData, int inh, int inw, int inchan,    int psize, int stride);

//ȫ��������
//������ �ò��Ȩ��ָ�룬ƫ��ָ�룬����ͼ������ָ�룬Ȩ��������Ȩ����������������
float* nn(float* weigth, float* bias, float* imgData, int weigthrow, int weigthcol, int activation); // NN����
//���ȫ���Ӳ����(�����Ҳ��soft)
//
char* intTochar(int i);

//// �����
//typedef struct convolutional_layer{
//	int inputWidth;   //����ͼ��Ŀ�
//	int inputHeight;  //����ͼ��ĳ�
//	int mapSize;      //����ģ��Ĵ�С��ģ��һ�㶼��������
//	int inChannels;   //����ͼ���ͨ����
//	int outChannels;  //���ͼ���ͨ����
//	int pad;          //padding��С
//	int stride;       //������С
//	// ��������ģ���Ȩ�طֲ���������һ����ά����
//	// ���СΪinChannels*outChannels*mapSize*mapSize��С
//	// ��������ά���飬��Ҫ��Ϊ�˱���ȫ���ӵ���ʽ��ʵ���Ͼ���㲢û���õ�ȫ���ӵ���ʽ
//	// �����������DeapLearningToolboox���CNN���ӣ����õ�����ȫ����
//	float**** mapData;     //�����ά����Ȩ�ص�����
//
//	float* basicData;   //ƫ�ã�ƫ�õĴ�С��ΪoutChannels
//	bool isFullConnect; //�Ƿ�Ϊȫ����
//	bool* connectModel; //����ģʽ��Ĭ��Ϊȫ���ӣ�  
//
//	// �������ߵĴ�Сͬ�����ά����ͬ
//	float*** v;			// �����ǰ����ά����ֵ
//	float*** y;			// ���������Ԫ����ά���
//
//}CovLayer;
//
//// ������ pooling
//typedef struct pooling_layer{
//	int inputWidth;   //����ͼ��Ŀ�
//	int inputHeight;  //����ͼ��ĳ�
//	int mapSize;      //����ģ��Ĵ�С
//	int inChannels;   //����ͼ�����Ŀ
//	int outChannels;  //���ͼ�����Ŀ
//
//	int poolType;     //Pooling�ķ���
//	float* basicData;   //ƫ��
//
//	float*** y; // ������������Ԫ����ά���,�޼����
//	//float*** d; // ����ľֲ��ݶ�,��ֵ
//}PoolLayer;
//
//// ����� ȫ���ӵ�������
//typedef struct nn_layer{
//	int inputNum;   //�������ݵ���Ŀ
//	int outputNum;  //������ݵ���Ŀ
//	float** wData; // ��άȨ�����ݣ�Ϊһ��inputNum*outputNum��С
//	float* basicData;   //ƫ�ã���СΪoutputNum��С
//
//	// �������ߵĴ�С�������ά����ͬ
//	float* v; // ���뼤���������ֵ
//	float* y; // ���������Ԫ�����
//	//float* d; // ����ľֲ��ݶ�,��ֵ
//
//	bool isFullConnect; //�Ƿ�Ϊȫ����
//}OutLayer;
//
////����ṹ
//typedef struct cnn_network{
//	int layerNum;    //������ܲ���
//	CovLayer* C1;     //�����ṹ�����C1
//	PoolLayer* S2;          //�ػ���ṹ�����S2
//	CovLayer* C3;     //�����ṹ�����C3
//	PoolLayer* S4;         //�ػ���ṹ�����S4
//	OutLayer* O5;    //ȫ���Ӳ�Ľṹ�����05
//	OutLayer* O6;    //�����ṹ�����06
//
//	//float* e; // ѵ�����
//	float* L; // ˲ʱ�������
//}CNN;
//
////ѵ������
//typedef struct train_opts{
//	int numepochs; // ѵ���ĵ�������
//	float alpha; // ѧϰ����
//}CNNOpts;


// cnn��ʼ������
//void cnnsetup(CNN* cnn,nSize inputSize,int outputSize);
/*	
	CNN�����ѵ������
	inputData��outputData�ֱ����ѵ������
	trainNum����������Ŀ
*/
//void cnntrain(CNN* cnn,	ImgArr inputData,LabelArr outputData,CNNOpts opts,int trainNum);
//
//// ����cnn����
//float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum);
//
//// ����cnn
//void savecnn(CNN* cnn, const char* filename);
//
//// ����cnn������
//void importcnn(CNN* cnn,  char* filename);

//// ��ʼ������㣬  7������ ���������ߡ�����˳ߴ硢����ͨ���������ͨ������padding��С��������С
//CovLayer* initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels,int pad, int stride);
//void CovLayerConnect(CovLayer* covL,bool* connectModel);
//
//// ��ʼ��������
//PoolLayer* initPoolLayer(int inputWidth,int inputHeigh,int mapSize,int inChannels,int outChannels,int poolType);
//void PoolLayerConnect(PoolLayer* poolL,bool* connectModel);
//
//// ��ʼ�������
//OutLayer* initOutLayer(int inputNum,int outputNum);

// ����� input�����ݣ�inputNum˵��������Ŀ��bas����ƫ��
float activation_Sigma(float input,float bas); // sigma�����

//void cnnff(CNN* cnn,float** inputData); // �����ǰ�򴫲�
//void cnnbp(CNN* cnn,float* outputData); // ����ĺ��򴫲�
//void cnnapplygrads(CNN* cnn,CNNOpts opts,float** inputData);
//void cnnclear(CNN* cnn); // ������vyd����

//�ϲ��ļ�mergeFile (infile1, infile2, filenmae)
void mergeFile(const char *fp1, const char *fp2);

/*
	Pooling Function
	input ��������
	inputNum ����������Ŀ
	mapSize ��ƽ����ģ������
*/
//void avgPooling(float** output,nSize outputSize,float** input,nSize inputSize,int mapSize); // ��ƽ��ֵ
/* 
	����ȫ����������Ĵ���
	nnSize������Ĵ�С
*/
//void nnff(float* output,float* input,float** wdata,float* bas,nSize nnSize); // ����ȫ�����������ǰ�򴫲�

//void savecnndata(CNN* cnn,const char* filename,float** inputdata); // ����CNN�����е��������

#endif
