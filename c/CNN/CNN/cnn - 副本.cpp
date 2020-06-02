#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"

void cnnsetup(CNN* cnn,nSize inputSize,int outputSize)
{
	cnn->layerNum=5;

	nSize mapSize;
	mapSize.w=inputSize.w;
	mapSize.h=inputSize.h;
	cnn->C1=initCovLayer(mapSize.w,mapSize.h,5,6,1,6);
	mapSize.w=mapSize.w-5+1;
	mapSize.h=mapSize.h-5+1;
	cnn->S2=initPoolLayer(mapSize.w,mapSize.h,2,6,6,AvePool);
	mapSize.w=mapSize.w/2;
	mapSize.h=mapSize.h/2;
	cnn->C3=initCovLayer(mapSize.w,mapSize.h,5,12,6,12);
	mapSize.w=mapSize.w-5+1;
	mapSize.h=mapSize.h-5+1;
	cnn->S4=initPoolLayer(mapSize.w,mapSize.h,2,12,12,AvePool);
	mapSize.w=mapSize.w/2;
	mapSize.h=mapSize.h/2;
	cnn->O5=initOutLayer(mapSize.w*mapSize.h*12,outputSize);

	cnn->e=(float*)calloc(cnn->O5->outputNum,sizeof(float));
}

CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int mapNum,int inChannels,int outChannels)
{
	CovLayer* covL=(CovLayer*)malloc(sizeof(CovLayer));

	covL->inputHeight=inputHeight;
	covL->inputWidth=inputWidth;
	covL->mapSize=mapSize;
	covL->mapNum=mapNum;
	covL->inChannels=inChannels;
	covL->outChannels=outChannels;

	covL->isFullConnect=true; // Ĭ��Ϊȫ����
	covL->DataSize=mapSize*mapSize*inChannels*outChannels;
	
	covL->mapData=(float*)malloc((covL->DataSize)*sizeof(float));
	covL->basicData=(float*)calloc(outChannels,sizeof(float));

	int outW=inputWidth-mapSize+1;
	int outH=inputHeight-mapSize+1;
	covL->d=(float*)calloc(outH*outW*outChannels,sizeof(float));
	covL->v=(float*)calloc(outH*outW*outChannels,sizeof(float));
	covL->y=(float*)calloc(outH*outW*outChannels,sizeof(float));

	// ����ģ��ĳ�ʼ��
	int i=0;
	srand((unsigned)time(NULL));
	for(i=0;i<covL->DataSize;i++){
		float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2; // ����һ��-1��1�������
		covL->mapData[i]=randnum*sqrt((float)6.0/(float)(mapSize*mapSize*(inChannels+outChannels)));
	}
	return covL;
}

PoolLayer* initPoolLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels,int poolType)
{
	PoolLayer* poolL=(PoolLayer*)malloc(sizeof(PoolLayer));

	poolL->inputHeight=inputHeight;
	poolL->inputWidth=inputWidth;
	poolL->mapSize=mapSize;
	poolL->inChannels=inChannels;
	poolL->outChannels=outChannels;
	poolL->poolType=poolType; 

	poolL->basicData=(float*)calloc(outChannels,sizeof(float));

	int outW=inputWidth/mapSize;
	int outH=inputHeight/mapSize;
	poolL->d=(float*)calloc(outH*outW*outChannels,sizeof(float));
	poolL->y=(float*)calloc(outH*outW*outChannels,sizeof(float));

	return poolL;
}

OutLayer* initOutLayer(int inputNum,int outputNum)
{
	OutLayer* outL=(OutLayer*)malloc(sizeof(OutLayer));

	outL->inputNum=inputNum;
	outL->outputNum=outputNum;

	outL->wData=(float*)malloc(inputNum*outputNum*sizeof(float));
	outL->basicData=(float*)calloc(outputNum,sizeof(float));

	outL->d=(float*)calloc(outputNum,sizeof(float));
	outL->v=(float*)calloc(outputNum,sizeof(float));
	outL->y=(float*)calloc(outputNum,sizeof(float));

	// Ȩ�صĳ�ʼ��
	int i=0;
	srand((unsigned)time(NULL));
	for(i=0;i<(inputNum*outputNum);i++){
		float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2; // ����һ��-1��1�������
		outL->wData[i]=randnum*sqrt((float)6.0/(float)(inputNum+outputNum));
	}

	outL->isFullConnect=true;

	return outL;
}

void cnntrain(CNN* cnn,nSize inputSize,int outputSize,
	float* inputData,float* outputData,int dataNum,CNNOpts opts)
{
	int e;
	for(e=0;e<opts.numepochs;e++){
		int n=0;
		for(n=0;n<dataNum;n++){
			float* indata=&inputData[n*inputSize.w*inputSize.h];
			float* outdata=&outputData[n*outputSize];

			cnnff(cnn,indata);  // ǰ�򴫲���������Ҫ�����
			cnnbp(cnn,outdata); // ���򴫲���������Ҫ�������Ԫ������ݶ�
			cnnapplygrads(cnn,opts); // ����Ȩ��

		}
	}
}

void cnnff(CNN* cnn,float* inputData)
{
	int outSizeW=cnn->S2->inputWidth;
	int outSizeH=cnn->S2->inputHeight;
	// ��һ��Ĵ���
	int i,j;
	// ��һ���������
	float** C1outData=(float**)malloc((cnn->S2->inChannels)*sizeof(float*));
	for(i=0;i<(cnn->C1->inChannels);i++)
		for(j=0;j<cnn->C1->outChannels;j++){
			float* map=&cnn->C1->mapData[(cnn->C1->mapSize*cnn->C1->mapSize)*j];
			nSize mapSize={cnn->C1->mapSize,cnn->C1->mapSize};
			nSize inSize={cnn->C1->inputWidth,cnn->C1->inputHeight};
			float bas=cnn->C1->basicData[j];

			float* outtemp=cov(map,mapSize,inputData,inSize);
			int k; // v�ĸ�ֵ
			for(k=0;k<outSizeW*outSizeH;k++)
				cnn->C1->v[j*outSizeH*outSizeW+k]=outtemp[k];

			C1outData[j]=activation_Sigma(outtemp,outSizeW*outSizeH,bas);
			for(k=0;k<outSizeW*outSizeH;k++) // y�ĸ�ֵ
				cnn->C1->y[j*outSizeH*outSizeW+k]=C1outData[j][k];
		}

	// �ڶ�����������
	outSizeH=cnn->S2->inputHeight/cnn->S2->mapSize;
	outSizeW=cnn->S2->inputWidth/cnn->S2->mapSize;
	float** S2outData=(float**)malloc((cnn->S2->outChannels)*sizeof(float*));
	for(i=0;i<(cnn->S2->outChannels);i++){
		float* input=C1outData[i];
		nSize inputSize={cnn->S2->inputWidth,cnn->S2->inputHeight};
		int mapSize=cnn->S2->mapSize;
		if(cnn->S2->poolType==AvePool)
			S2outData[i]=avgPooling(input,inputSize,mapSize);

		int k;
		for(k=0;k<outSizeW*outSizeH;k++) // y�ĸ�ֵ
			cnn->S2->y[i*outSizeH*outSizeW+k]=S2outData[i][k];
	}

	// �������������,������ȫ����
	outSizeW=cnn->S4->inputWidth;
	outSizeH=cnn->S4->inputHeight;
	float** C3outData=(float**)malloc((cnn->S4->inChannels)*sizeof(float*));
	for(j=0;j<cnn->C3->outChannels;j++){
		float* z=(float*)calloc(outSizeW*outSizeH,sizeof(float));
		float* map=&cnn->C3->mapData[(cnn->C3->mapSize*cnn->C3->mapSize)*j];
		nSize mapSize={cnn->C3->mapSize,cnn->C3->mapSize};
		nSize inSize={cnn->C3->inputWidth,cnn->C3->inputHeight};
		float bas=cnn->C3->basicData[j];

		for(i=0;i<(cnn->C3->inChannels);i++){
			float* outtemp=cov(map,mapSize,S2outData[i],inSize);
			int s;
			for(s=0;s<(outSizeH*outSizeW);s++)
				z[s]=z[s]+outtemp[s];
		}

		int k; // v�ĸ�ֵ
		for(k=0;k<outSizeW*outSizeH;k++)
			cnn->C3->v[j*outSizeH*outSizeW+k]=z[k];

		C3outData[j]=activation_Sigma(z,outSizeW*outSizeH,bas);

		for(k=0;k<outSizeW*outSizeH;k++) // y�ĸ�ֵ
			cnn->C3->y[j*outSizeH*outSizeW+k]=C3outData[j][k];
	}

	// ���Ĳ���������
	outSizeH=cnn->S4->inputHeight/cnn->S4->mapSize;
	outSizeW=cnn->S4->inputWidth/cnn->S4->mapSize;
	float** S4outData=(float**)malloc((cnn->S4->outChannels)*sizeof(float*));
	for(i=0;i<(cnn->S4->outChannels);i++){
		float* input=C3outData[i];
		nSize inputSize={cnn->S4->inputWidth,cnn->S4->inputHeight};
		int mapSize=cnn->S4->mapSize;
		if(cnn->S4->poolType==AvePool)
			S4outData[i]=avgPooling(input,inputSize,mapSize);

		int k;
		for(k=0;k<outSizeW*outSizeH;k++) // y�ĸ�ֵ
			cnn->S4->y[i*outSizeH*outSizeW+k]=S4outData[i][k];
	}

	// �����Ĵ���
	float* O5inData=(float*)malloc((cnn->O5->inputNum)*sizeof(float)); //չ���õ�����������
	int blocksize=cnn->O5->inputNum/(cnn->S4->outChannels);
	for(i=0;i<(cnn->S4->outChannels);i++)
		for(j=0;j<blocksize;j++)
			O5inData[i*blocksize+j]=S2outData[i][j];

	nSize nnSize={cnn->O5->inputNum,cnn->O5->outputNum};
	float* z=nnff(O5inData,cnn->O5->wData,cnn->O5->basicData,nnSize);
	int k; // v�ĸ�ֵ
	for(k=0;k<nnSize.h;k++)
		cnn->O5->v[k]=z[k];

	float* O5outData=activation_Sigma(z,nnSize.h,0);
	for(k=0;k<nnSize.h;k++)
		cnn->O5->y[k]=O5outData[k];
	free(z);

	// �ͷ���ض�̬����
	// C1outData
	for(i=0;i<(cnn->S2->inChannels);i++){
		float* temp=C1outData[i];
		free(temp);
		temp=NULL;
	}
	free(C1outData);
	// S2outData
	for(i=0;i<(cnn->S2->outChannels);i++){
		float* temp=S2outData[i];
		free(temp);
		temp=NULL;
	}
	free(S2outData);
	// C3outData
	for(i=0;i<(cnn->S4->inChannels);i++){
		float* temp=C3outData[i];
		free(temp);
		temp=NULL;
	}
	free(C3outData);
	// S4outData
	for(i=0;i<(cnn->S4->outChannels);i++){
		float* temp=S4outData[i];
		free(temp);
		temp=NULL;
	}
	free(S4outData);
	// O5inData
	free(O5inData);
}

float* cov(float* map,nSize mapSize,float* inputData,nSize inSize) // �������
{
	int outSizeW=inSize.w-mapSize.w+1;
	int outSizeH=inSize.h-mapSize.h+1;
	float* outputData=(float*)calloc(outSizeW*outSizeH,sizeof(float));
	int i,j;
	int m,n;// ���ģ��Ĵ�С
	int halfmapsize=(mapSize.w-1)/2; // ���ģ��İ���С
	for(i=0;i<outSizeW;i++)
		for(j=0;j<outSizeH;j++)
			for(m=0;m<mapSize.w;m++)
				for(n=0;n<mapSize.h;n++){
					outputData[i+j*outSizeW]=outputData[i+j*outSizeW]+map[m+n*mapSize.w]*
						inputData[i+halfmapsize-m+(j+halfmapsize-n)*inSize.w];
				}
	return outputData;
}

// ����� input�����ݣ�inputNum˵��������Ŀ��bas����ƫ��
float* activation_Sigma(float* input,int inputNum,float bas) // sigma�����
{
	int i;
	float* output=(float*)malloc(inputNum*sizeof(float));
	for(i=0;i<inputNum;i++){
		float temp=input[i]+bas;
		output[i]=(float)1.0/((float)(1.0+exp(-temp)));
	}
	return output;
}

float* avgPooling(float* input,nSize inputSize,int mapSize) // ��ƽ��ֵ
{
	int outputW=inputSize.w/mapSize;
	int outputH=inputSize.h/mapSize;
	float* output=(float*)malloc((outputW*outputH)*sizeof(float));
	int i,j;
	int m,n;
	for(i=0;i<outputW;i++)
		for(j=0;j<outputH;j++)
		{
			float sum=0;
			for(m=i;m<i+mapSize;m++)
				for(n=j;n<j+mapSize;n++)
					sum=sum+input[m+n*inputSize.w];

			output[i+j*outputW]=sum/(mapSize*mapSize);
		}

	return output;
}

// ����ȫ�����������ǰ�򴫲�
float vecMulti(float* vec1,float* vec2,int vecL)// ���������
{
	int i;
	float m=0;
	for(i=0;i<vecL;i++)
		m=m+vec1[i]*vec2[i];
	return m;
}
float* nnff(float* input,float* wdata,float* bas,nSize nnSize)
{
	int w=nnSize.w;
	int h=nnSize.h;
	float* z=(float*)malloc(h*sizeof(float));
	int i;
	for(i=0;i<h;i++){
		float* wi=&wdata[w*i];
		z[i]=vecMulti(input,wi,w)+bas[i];
	}
	return z;
}

float sigma_derivation(float y){ // Logic��������Ա���΢��
	return y*(1-y); // ����y��ָ��������������ֵ���������Ա���
}

float* vecExpand(float* vec,nSize vecSize,int addw,int addh){ // ���������䣬������addw����������addh��
	int w=vecSize.w;
	int h=vecSize.h;
	float* res=(float*)malloc(w*h*addw*addh*sizeof(float));
	
	int i,j,m,n;
	for(j=0;j<h*addh;j=j+addh){
		for(i=0;i<w*addw;i=i+addw)// �������
			for(m=0;m<addw;m++)
				res[i+m+j*w*addw]=vec[i/addw+j*w/addw];

		for(n=1;n<addh;n++)      //  �ߵ�����
			for(i=0;i<w*addw;i++)
				res[i+(j+n)*w*addw]=res[i+j*w*addw];
	}
	return res;
}
float* vecEdgeExpand(float* vec,nSize vecSize,int addw,int addh){ // ������Ե����
	int w=vecSize.w;
	int h=vecSize.h;
	float* res=(float*)malloc((w+2*addw)*(h+2*addh)*sizeof(float));

	int i,j,m,n;
	for(j=0;j<h+2*addh;j++){
		for(i=0;i<w+2*addw;i++){
			if(j<addh||i<addw||j>=(h+addh)||i>=(w+addw))
				res[i+j*(w+2*addw)]=0.0;
			else
				res[i+j*(w+2*addw)]=vec[i-addw+(j-addh)*w]; // ����ԭ����������
		}
	}
	return res;
}
float* vecEdgeShrink(float* vec,nSize vecSize,int shrinkw,int shrinkh){ // ��������С������Сaddw������Сaddh
	int w=vecSize.w;
	int h=vecSize.h;
	float* res=(float*)malloc((w-2*shrinkw)*(h-2*shrinkh)*sizeof(float));

	int i,j;
	for(j=0;j<h;j++){
		for(i=0;i<w;i++){
			if(j>=shrinkh&&i>=shrinkw&&j<(h-shrinkh)&&i<(w-shrinkw))
				res[(i-shrinkw)+(j-shrinkh)*(w-2*shrinkw)]=vec[i+j*w]; // ����ԭ����������
		}
	}
	return res;
}

float* correlation(float* map,nSize mapSize,float* inputData,nSize inSize)// �����
{
	// ����Ļ�������ں��򴫲�ʱ���ã������ڽ�Map��ת180���پ��
	// Ϊ�˷�����㣬�����Ƚ�ͼ������һȦ
	int halfmapsizew=(mapSize.w-1)/2; // ���ģ��İ���С
	int halfmapsizeh=(mapSize.h-1)/2;
	int outSizeW=inSize.w+(mapSize.w-1)*2; // ������������һ����
	int outSizeH=inSize.h+(mapSize.h-1)*2;
	float* outputData=(float*)calloc(outSizeW*outSizeH,sizeof(float)); // ����صĽ��������

	// Ϊ�˷�����㣬��inputData����һȦ
	float* exInputData=vecEdgeExpand(inputData,inSize,mapSize.w-1,mapSize.h-1);

	int i,j;
	int m,n;// ���ģ��Ĵ�С
	
	for(i=halfmapsizew;i<outSizeW-halfmapsizew;i++)
		for(j=halfmapsizeh;j<outSizeH-halfmapsizeh;j++)
			for(m=0;m<mapSize.w;m++)
				for(n=0;n<mapSize.h;n++){
					outputData[i+j*outSizeW]=outputData[i+j*outSizeW]+map[m+n*mapSize.w]*
						exInputData[i-halfmapsizew+m+(j-halfmapsizeh+n)*outSizeW];
				}

	nSize outSize={outSizeW,outSizeH};
	return vecEdgeShrink(outputData,outSize,halfmapsizew,halfmapsizeh);
}

void cnnbp(CNN* cnn,float* outputData) // ����ĺ��򴫲�
{
	int i,j,m,n; // �����浽������
	for(i=0;i<cnn->O5->outputNum;i++)
		cnn->e[i]=outputData[i]-cnn->O5->y[i];

	/*�Ӻ���ǰ�������*/
	// �����O5
	int dnum=cnn->O5->outputNum;
	for(i=0;i<dnum;i++)
		cnn->O5->d[i]=cnn->e[i]*sigma_derivation(cnn->O5->y[i]);

	// S4�㣬���ݵ�S4������
	// ����û�м����
	for(i=0;i<cnn->O5->inputNum;i++)
		for(j=0;j<cnn->O5->outputNum;j++)
			cnn->S4->d[i]=cnn->S4->d[i]+cnn->O5->d[j]*cnn->O5->wData[j*cnn->O5->inputNum+i];

	// C3��
	// ��S4�㴫�ݵĸ��������,����ֻ����S4���ݶ�������һ��
	int outW=cnn->S4->inputHeight*cnn->S4->inputWidth;
	int mapdata=cnn->S4->mapSize;
	nSize S4dSize={cnn->S4->inputWidth/mapdata,cnn->S4->inputHeight*cnn->S4->outChannels/mapdata};
	// �����Pooling����ƽ�������Է��򴫵ݵ���һ��Ԫ������ݶ�û�б仯
	float* C3e=vecExpand(cnn->S4->d,S4dSize,mapdata,mapdata);

	for(i=0;i<outW;i++)
		cnn->C3->d[i]=C3e[i]*sigma_derivation(cnn->C3->y[i]);

	// S2�㣬S2��û�м����������ֻ�о�����м��������
	// �ɾ���㴫�ݸ������������ݶȣ��������㹲��6*12�����ģ��

	nSize mapSize={cnn->C3->mapSize,cnn->C3->mapSize};
	int mapDataW=cnn->C3->mapSize*cnn->C3->mapSize*cnn->C3->outChannels;
	int mapDataSize=cnn->C3->mapSize*cnn->C3->mapSize;
	nSize inputSize={cnn->S4->inputWidth,cnn->S4->inputHeight};
	for(i=0;i<cnn->C3->inChannels;i++){
		float* S2e=(float*)calloc(cnn->C3->inputHeight*cnn->C3->inputWidth,sizeof(float));
		int k;
		for(j=0;j<cnn->C3->outChannels;j++){
			float* e=correlation(&cnn->C3->mapData[j*mapDataSize+i*mapDataW],mapSize,
				&cnn->C3->d[i*(cnn->C3->inputHeight*cnn->C3->outChannels)],inputSize);
			
			for(k=0;k<cnn->C3->inputHeight*cnn->C3->inputWidth;k++)
				S2e[k]=S2e[k]+e[k];
		}
		for(k=0;k<cnn->C3->inputHeight*cnn->C3->inputWidth;k++)
			cnn->S2->d[i*cnn->C3->inputHeight*cnn->C3->inputWidth+k]=S2e[k]*
			sigma_derivation(cnn->S2->y[i*cnn->C3->inputHeight*cnn->C3->inputWidth+k]);
	}

	// C1�㣬�����
	outW=cnn->S2->inputHeight*cnn->S2->inputWidth;
	mapdata=cnn->S2->mapSize;
	nSize S2dSize={cnn->S2->inputWidth/mapdata,cnn->S2->inputHeight*cnn->S2->outChannels/mapdata};
	float* C1e=vecExpand(cnn->S2->d,S2dSize,mapdata,mapdata);

	for(i=0;i<cnn->C1->outChannels;i++){
		for(j=0;j<outW;j++)
		cnn->C1->d[i*outW+j]=C1e[i*outW+j]*sigma_derivation(cnn->C1->y[i*outW+j]);
	}	
}

float* flipall(float* matVec, nSize matSize) // ���������ĵ㷭ת
{
	int w=matSize.w;
	int h=matSize.h;
	float* res=(float*)calloc(w*h,sizeof(float));
	int i,j;
	for(i=0;i<w;i++)
		for(j=0;j<h;j++)
			res[i+j*w]=matVec[(w-i-1)+(h-j-1)*w];

	return res;
}

void cnnapplygrads(CNN* cnn,CNNOpts opts) // ����Ȩ��
{
	// �������Ȩ�ص���Ҫ�Ǿ����������
	// �����������ط���Ȩ�ؾͿ�����
	int i,j;

	// C1���Ȩ�ظ���
	int outW=cnn->S2->inputHeight*cnn->S2->inputWidth;
	nSize dSize={cnn->S2->inputHeight,cnn->S2->inputWidth};
	int iny=cnn->C1->inputHeight*cnn->C1->inputWidth;
	nSize ySize={cnn->C1->inputHeight,cnn->C1->inputWidth};

	for(i=0;i<cnn->C1->outChannels;i++){
		float* dw=cov(&cnn->C1->d[i*outW],dSize,flipall(cnn->C1->y,ySize),ySize); // wȨ���ݶȱ仯
		int wSize=cnn->C1->mapSize*cnn->C1->mapSize;
		for(j=0;j<wSize;j++)
			cnn->C1->mapData[i*wSize+j]=cnn->C1->mapData[i*wSize+j]-opts.alpha*dw[j]; // ����Ȩ��

		float db=0.0;
		for(j=0;j<outW;j++)
			db=db+cnn->C1->d[i*outW+j];
		cnn->C1->basicData[i]=cnn->C1->basicData[i]-opts.alpha*db;
	}

	// C3���Ȩ�ظ���
	outW=cnn->S4->inputHeight*cnn->S4->inputWidth;
	dSize.h=cnn->S4->inputHeight;
	dSize.w=cnn->S4->inputWidth;
	iny=cnn->C3->inputHeight*cnn->C3->inputWidth;
	ySize.h=cnn->C3->inputHeight;
	ySize.w=cnn->C3->inputWidth;

	int n;
	for(n=0;n<cnn->C3->inChannels;n++){
		for(i=0;i<cnn->C3->outChannels;i++){
			float* dw=cov(&cnn->C3->d[i*outW],dSize,flipall(&cnn->C3->y[n*iny],ySize),ySize); // wȨ���ݶȱ仯
			int wSize=cnn->C3->mapSize*cnn->C3->mapSize;
			for(j=0;j<wSize;j++)
				cnn->C3->mapData[n*wSize*cnn->C3->outChannels+i*wSize+j]=
				cnn->C3->mapData[n*wSize*cnn->C3->outChannels+i*wSize+j]-opts.alpha*dw[j]; // ����Ȩ��

			float db=0.0;
			for(j=0;j<outW;j++)
				db=db+cnn->C3->d[i*outW+j];
			cnn->C3->basicData[n*cnn->C3->outChannels+i]=cnn->C3->basicData[n*cnn->C3->outChannels+i]-opts.alpha*db;
		}
	}

	// �����
	for(j=0;j<cnn->O5->outputNum;j++){
		for(i=0;i<cnn->O5->inputNum;i++)
			cnn->O5->wData[j*cnn->O5->inputNum+i]=cnn->O5->wData[j*cnn->O5->inputNum+i]-
			opts.alpha*cnn->O5->d[j]*cnn->S4->y[i];
		cnn->O5->basicData[j]=cnn->O5->basicData[j]-opts.alpha*cnn->O5->d[j];
	}	
}
