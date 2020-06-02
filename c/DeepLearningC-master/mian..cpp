#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"
#include "minst.h"


/*主函数*/
int main()
{	
	//导入测试图片
	MinstImg testImg = read_Img("D:\\FPGA-AI\\Lenet_MNIST\\DeepLearningC-master\\CNN\\CNNdata\\input_struct11_int32_whc_6_8.bin");
	//LabelArr testLabel = read_Lable("D:\\FPGA-AI\\Lenet_MNIST\\DeepLearningC-master\\CNN\\data\\test-labels-idx1-ubyte.gz");
	
	//从输入图片中获取输入的宽、高
	int input_w = testImg.h;
	int input_h = testImg.w;
	int input_ch = testImg.ch;
	printf("导入图像成功");
	//int outSize=testLabel->LabelPtr[0].l;                        //从输出标签中获取输出维度（10） 

	// CNN结构的初始化，为网络中每层的变量、数据分配内存空间
	CNN* cnn=(CNN*)malloc(sizeof(CNN));

	/*

	//权重的基地址
	char* basedir = "D:\\FPGA-AI\\Lenet_MNIST\\DeepLearningC-master\\CNN\\weight_bin\\\\";
	
	const char* c1w_filename = combine_strings(basedir, "c1w.bin");   //c1卷积权重路径
	//mergeFile(c1w_filename, allweigth_filename);
	
	const char* c1b_filename = combine_strings(basedir, "c1b.bin");   //c1偏置路径
	const char* c3w_filename = combine_strings(basedir, "c2w.bin");	  //c2卷积权重路径
	const char* c3b_filename = combine_strings(basedir, "c2b.bin");   //c2偏置路径
	const char* f5w_filename = combine_strings(basedir, "f6w.bin");   //fc6权重路径
	const char* f5b_filename = combine_strings(basedir, "f6b.bin");   //fc6偏置路径
	const char* f6w_filename = combine_strings(basedir, "f8w.bin");   //fc8权重路径
	const char* f6b_filename = combine_strings(basedir, "f8b.bin");   //fc8偏置路径

	// 网络前向传播
	// cnnff(cnn,inputData->ImgPtr[n].ImgData);    //对第n张图片的数据进行前向传播
	// 这里InputData是图像数据，inputData[r][c],r行c列，这里根各权重模板是一致的

		// 第一层的传播
		int outSizeW = cnn->S2->inputWidth;     //c1层的输出宽
		int outSizeH = cnn->S2->inputHeight;   //c1层的输出高

		int i, j, r, c;
		// 第一层输出数据
		nSize mapSize = { cnn->C1->mapSize, cnn->C1->mapSize };          //C1层的卷积核尺寸大小
		nSize inSize = { cnn->C1->inputWidth, cnn->C1->inputHeight };    //c1层的输入宽、高
		nSize outSize = { cnn->S2->inputWidth, cnn->S2->inputHeight };   //c1层的输出宽、高

		conv(cnn->C1, inputData);

		
		for(i=0;i<(cnn->C1->outChannels);i++){           //遍历c1层的输入通道
		for(j=0;j<(cnn->C1->inChannels);j++){           //遍历c1层的输出通道
		//参数：c1的一张卷积核权重、卷积核大小、输入图像数据（二进制）、输入的宽高、空
		float** mapout=conv(cnn->C1->mapData[j][i],mapSize,inputData,inSize,valid);
		//添加偏置
		addmat(cnn->C1->v[i],cnn->C1->v[i],outSize,mapout,outSize);
		for(r=0;r<outSize.r;r++)
		free(mapout[r]);
		free(mapout);
		}
		for(r=0;r<outSize.r;r++)
		for(c=0;c<outSize.c;c++)
		cnn->C1->y[i][r][c]=activation_Sigma(cnn->C1->v[i][r][c],cnn->C1->basicData[i]);
		}
		


		// 第二层的输出传播S2，采样层
		outSize.c = cnn->C3->inputWidth;
		outSize.r = cnn->C3->inputHeight;
		inSize.c = cnn->S2->inputWidth;
		inSize.r = cnn->S2->inputHeight;
		for (i = 0; i<(cnn->S2->outChannels); i++){
			if (cnn->S2->poolType == AvePool)
				avgPooling(cnn->S2->y[i], outSize, cnn->C1->y[i], inSize, cnn->S2->mapSize);
		}

		// 第三层输出传播,卷积连接
		outSize.c = cnn->S4->inputWidth;
		outSize.r = cnn->S4->inputHeight;
		inSize.c = cnn->C3->inputWidth;
		inSize.r = cnn->C3->inputHeight;
		mapSize.c = cnn->C3->mapSize;
		mapSize.r = cnn->C3->mapSize;
		for (i = 0; i<(cnn->C3->outChannels); i++){
			for (j = 0; j<(cnn->C3->inChannels); j++){
				float** mapout = cov(cnn->C3->mapData[j][i], mapSize, cnn->S2->y[j], inSize, valid);
				addmat(cnn->C3->v[i], cnn->C3->v[i], outSize, mapout, outSize);
				for (r = 0; r<outSize.r; r++)
					free(mapout[r]);
				free(mapout);
			}
			for (r = 0; r<outSize.r; r++)
				for (c = 0; c<outSize.c; c++)
					cnn->C3->y[i][r][c] = activation_Sigma(cnn->C3->v[i][r][c], cnn->C3->basicData[i]);
		}

		// 第四层的输出传播
		inSize.c = cnn->S4->inputWidth;
		inSize.r = cnn->S4->inputHeight;
		outSize.c = inSize.c / cnn->S4->mapSize;
		outSize.r = inSize.r / cnn->S4->mapSize;
		for (i = 0; i<(cnn->S4->outChannels); i++){
			if (cnn->S4->poolType == AvePool)
				avgPooling(cnn->S4->y[i], outSize, cnn->C3->y[i], inSize, cnn->S4->mapSize);
		}

		// 输出层O5的处理
		// 首先需要将前面的多维输出展开成一维向量
		float* O5inData = (float*)malloc((cnn->O5->inputNum)*sizeof(float));
		for (i = 0; i<(cnn->S4->outChannels); i++)
			for (r = 0; r<outSize.r; r++)
				for (c = 0; c<outSize.c; c++)
					O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = cnn->S4->y[i][r][c];

		nSize nnSize = { cnn->O5->inputNum, cnn->O5->outputNum };
		nnff(cnn->O5->v, O5inData, cnn->O5->wData, cnn->O5->basicData, nnSize);
		for (i = 0; i<cnn->O5->outputNum; i++)
			cnn->O5->y[i] = activation_Sigma(cnn->O5->v[i], cnn->O5->basicData[i]);
		free(O5inData);


	
	//前向传播，得到网络输出的结果就行
	int testNum=1;     //测试数目先设为1 
	float incorrectRatio=0.0;     //分类识别错误率
	

	incorrectRatio=cnntest(cnn,testImg,testLabel,testNum);    //预测，(不需要标签)，得出分类识别错误率
	printf("testing result: %f\n",incorrectRatio );    //打印测试结果
	printf("test finished!!\n");  
	*/
	//return 0;
	system("pause");

	
}