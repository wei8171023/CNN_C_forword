#ifndef __CNN_
#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "minst.h"

#define AvePool 0
#define MaxPool 1
#define MinPool 2

// �����
typedef struct convolutional_layer{
	int inputWidth;   //����ͼ��Ŀ�
	int inputHeight;  //����ͼ��ĳ�
	int mapSize;      //����ģ��Ĵ�С
	int mapNum;       //����ģ�����Ŀ
	int inChannels;   //����ͼ�����Ŀ
	int outChannels;  //���ͼ�����Ŀ

	int DataSize;
	float* mapData;     //�������ģ�������
	float* basicData;   //ƫ��
	bool isFullConnect; //�Ƿ�Ϊȫ����
	bool* connectModel; //����ģʽ��Ĭ��Ϊȫ���ӣ�

	// �������ߵĴ�Сͬ�����ά����ͬ
	float* v; // ���뼤���������ֵ
	float* y; // ���������Ԫ�����
	float* d; // ����ľֲ��ݶ�,��ֵ  
}CovLayer;

// ������ pooling
typedef struct pooling_layer{
	int inputWidth;   //����ͼ��Ŀ�
	int inputHeight;  //����ͼ��ĳ�
	int mapSize;      //����ģ��Ĵ�С

	int inChannels;   //����ͼ�����Ŀ
	int outChannels;  //���ͼ�����Ŀ

	int poolType;     //Pooling�ķ���
	float* basicData;   //ƫ��
	bool isFullConnect; //�Ƿ�Ϊȫ����
	bool* connectModel; //����ģʽ��Ĭ��Ϊȫ���ӣ�

	float* y; // ������������Ԫ�����,�޼����
	float* d; // ����ľֲ��ݶ�,��ֵ
}PoolLayer;

// ����� ȫ���ӵ�������
typedef struct nn_layer{
	int inputNum;   //�������ݵ���Ŀ
	int outputNum;  //������ݵ���Ŀ

	float* wData; // Ȩ�����ݣ�Ϊһ��inputNum*outputNum��С
	float* basicData;   //ƫ�ã���СΪoutputNum��С

	// �������ߵĴ�Сͬ�����ά����ͬ
	float* v; // ���뼤���������ֵ
	float* y; // ���������Ԫ�����
	float* d; // ����ľֲ��ݶ�,��ֵ

	bool isFullConnect; //�Ƿ�Ϊȫ����
}OutLayer;

typedef struct cnn_network{
	int layerNum;
	CovLayer* C1;
	PoolLayer* S2;
	CovLayer* C3;
	PoolLayer* S4;
	OutLayer* O5;

	float* e; // ѵ�����
}CNN;

typedef struct ImgSize{
	int w;
	int h;
}nSize;

typedef struct train_opts{
	int numepochs; // ѵ���ĵ�������
	float alpha; // ѧϰ����
}CNNOpts;

void cnnsetup(CNN* cnn,nSize inputSize,int outputSize);
/*	
	CNN�����ѵ������
	inputData��outputData�ֱ����ѵ������
	dataNum����������Ŀ
*/
void cnntrain(CNN* cnn,nSize inputSize,int outputSize,
	float* inputData,float* outputData,int dataNum,CNNOpts opts);

// ��ʼ�������
CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int mapNum,int inChannels,int outChannels);
void CovLayerConnect(CovLayer* covL,bool* connectModel);
// ��ʼ��������
PoolLayer* initPoolLayer(int inputWidth,int inputHeigh,int mapSize,int inChannels,int outChannels,int poolType);
void PoolLayerConnect(PoolLayer* poolL,bool* connectModel);
// ��ʼ�������
OutLayer* initOutLayer(int inputNum,int outputNum);

// ����� input�����ݣ�inputNum˵��������Ŀ��bas����ƫ��
float* activation_Sigma(float* input,int inputNum,float bas); // sigma�����

void cnnff(CNN* cnn,float* inputData); // �����ǰ�򴫲�
void cnnbp(CNN* cnn,float* outputData); // ����ĺ��򴫲�
void cnnapplygrads(CNN* cnn,CNNOpts opts);

float* cov(float* map,nSize mapSize,float* inputData,nSize inSize); // �������

/*
	Pooling Function
	input ��������
	inputNum ����������Ŀ
	mapSize ��ƽ����ģ������
*/
float* avgPooling(float* input,nSize inputSize,int mapSize); // ��ƽ��ֵ

/* 
	����ȫ����������Ĵ���
	nnSize������Ĵ�С
*/
float* nnff(float* input,float* wdata,float* bas,nSize nnSize); // ����ȫ�����������ǰ�򴫲�

#endif
