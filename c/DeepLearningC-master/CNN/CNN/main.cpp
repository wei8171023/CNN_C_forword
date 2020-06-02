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
	//导入测试图片到图像结构体Img
	printf("开始导入图像.....\n");
	MinstImg Img = read_Img("D:\\FPGA-AI\\Lenet_MNIST\\DeepLearningC-master\\CNN\\CNNdata\\input_struct11_float32_whc_6_8.bin"); 
	printf("导入图像成功!\n");
	printf("\n");
	//int l, m, n;
	//for (l= 0; l < 1; l++)    //遍历输出通道==卷积核个数
	//	for (m = 0; m < 28; m++)         //遍历输出行
	//		for (n= 0; n< 28; n++)          //遍历输出列
	//			printf("%f ", Img.ImgData[l * 28 * 28 + m * 28 + n]);
	//	printf("\n");
	

	//权重的基地址
	char* basedir = "D:\\FPGA-AI\\Lenet_MNIST\\DeepLearningC-master\\CNN\\weight_bin\\";
	//定义输出的特征图
	float* out;
	/*
	conv1(rule激活)，返回输出特征图out
	*/
	//从输入图片中获取输入的宽、高
	int imgw = Img.h;           
	int imgh = Img.w;            
	int imgch = Img.ch;      
	//导入c1权重
	//导入c1卷积权重到1维float型指针c1w中,导入c1偏置到1维float型指针c1b中
	printf("开始导入c1卷积权重.....\n");
	const char* c1w_filename = combine_strings(basedir, "c1w.bin");   //c1卷积权重路径
	float* c1w = read_weight(c1w_filename,3,1,16);      //卷积权重文件 、卷积核尺寸、卷积核个数   
	printf("导入c1卷积权重成功!\n");
	//导入c1偏置
	printf("开始导入c1偏置.....\n");
	const char* c1b_filename = combine_strings(basedir, "c1b.bin");   //c1偏置路径
	float* c1b = read_bias(c1b_filename,16);            //偏置文件、偏置元素个数    
	printf("导入c1偏置成功!\n");
	//计时开始
	clock_t t1, t2;
	t1 = clock();
	printf("开始conv1计算..... \n");
	//求conv1
	//参数：该层的权重指针，输入图像数据指针、输入宽、输入高、输入通道数、输出通道数、卷积核的大小、步幅大小、补边否	
	out = conv(c1w,c1b,Img.ImgData,28,28,1,16,3,1,1,1);
	//计时结束
	t2 = clock();
	
	printf("conv1计算完成,耗时:%20.10f ms! \n", (double)(t2 - t1) / CLOCKS_PER_SEC*1000.0);
	printf("\n");

	/*
	maxpool1,返回输出特征图out
	*/
	clock_t t3, t4;
	t3 = clock();
	printf("开始pool计算.....  \n");
	out = maxpool(out, 28, 28, 16, 2, 2);
	t4 = clock();
	printf("pool1计算完成,耗时:%20.10f ms! \n", (double)(t4 - t3) / CLOCKS_PER_SEC*1000.0);
	printf("\n");

	/*
	conv2(relu激活),返回输出特征图out
	*/
	//导入c2卷积权重到1维float型指针c2w中,导入c2偏置到1维float型指针c2b中
	const char* c2w_filename = combine_strings(basedir, "c2w.bin");   //c1卷积权重路径
	float* c2w = read_weight(c2w_filename, 3, 16, 32);      //卷积权重文件 、卷积核尺寸、卷积核个数   
	printf("导入c2卷积权重成功!\n");
	//导入c2偏置
	const char* c2b_filename = combine_strings(basedir, "c2b.bin");   //c1偏置路径
	float* c2b = read_bias(c2b_filename, 32);    //偏置文件、偏置元素个数    
	printf("导入c2偏置成功!\n");
	//求c2卷积和激活
	//参数：该层的权重指针，输入图像数据指针、输入宽、输入高、输入通道数、卷积核的个数、卷积核的大小、步幅大小、补边否
	//int pad=(fsize-1)/2;
	clock_t t5, t6;
	t5 = clock();
	printf("开始conv2计算...... \n");
	out = conv(c2w, c2b, out, 14, 14, 16, 32, 3, 1, 1,1);
	t6 = clock();
	printf("conv2计算完成！耗时:%20.10f ms! \n", (double)(t6 - t5) / CLOCKS_PER_SEC*1000.0);
	printf("\n");
	/*
	maxpool2,返回输出特征图out
	*/
	clock_t t7, t8;
	t7 = clock();
	printf("开始poo2计算.....  \n");
	out = maxpool(out, 14, 14, 32, 2, 2);
	t8 = clock();
	printf("pool2计算完成,耗时:%20.10f ms! \n", (double)(t8 - t7) / CLOCKS_PER_SEC*1000.0);
	printf("\n");

	printf("查验pool2的计算结果！  \n");
	int i, j, k;
	//for (k = 0; k < 32; k++)    //遍历输出通道==卷积核个数
	//{
	//	for (i = 0; i < 7; i++)         //遍历输出行
	//	{
	//		for (j = 0; j < 7; j++)          //遍历输出列
	//		{
	//			printf("%f  ", out[k * 7 * 7 + i * 7 + j]);
	//		}
	//	}
	//	printf("\n");
	//}

	/*
	flatten   将池化输出的结果（32,7,7）cfirst排列顺序，转置为（7,7，32）clast模式，以适应keras中权重和数据的clast顺序。
	*/
	printf("开始flatten \n");
	float** v = (float**)malloc(32 * sizeof(float*));      //为转置前的二维指针v分配内存空间[32][7*7]
	for (i = (32- 1); i != (-1); i--)
	{
		v[i] = (float*)calloc(49, sizeof(float));
	}
	//float* y = (float*)calloc(1568, sizeof(float));

	float** y = (float**)malloc(49 * sizeof(float*));      //为转置后的二维指针y分配内存空间[7*7][32]
	for (i = (49 - 1); i != (-1); i--)
	{
		y[i] = (float*)calloc(32, sizeof(float));
	}
	//float* y = (float*)calloc(1568, sizeof(float));

	int r, c;
	int l = 0;
	for (j = 0; j < 32; j++)//16     //3维(1维)转2维
	{

		for (r = 0; r < 7; r++)
		{
			for (c = 0; c < 7; c++)
			{
				v[j][l] = out[j*49+r*7+c];
				l++;
			}
		}
		l = 0;//开始下一行复制
	}
	printf("一维转二维完成！ \n");

	int p, d;
	float temp = 0.0;
	for (i = 0; i < 32; i++)         //2维数组v转置，到2维数组y
	{
		for (int j = 0; j < 49; j++)
		{
			y[j][i] = v[i][j];
		}
	}
	printf("二维v转置完成！ \n");

	k = 0;
	for (int p = 0; p < 49; p++)      //2维y转到1维out中
	{
		for (int d = 0; d < 32; d++)
		{

			out[k] = y[p][d];
			k++;
		}
	}
	printf("二维v转一维out完成！ \n");

	//printf("flatten后的结果！  \n");
	//for (k = 0; k < 32; k++)    //遍历输出通道==卷积核个数
	//{
	//	for (i = 0; i < 7; i++)         //遍历输出行
	//	{
	//		for (j = 0; j < 7; j++)          //遍历输出列
	//		{
	//			printf("%f  ", out[k * 7 * 7 + i * 7 + j]);
	//		}
	//	}
	//	printf("\n");
	//}


	///*
	//fc1(relu激活)
	//*/
	////导入fc1卷积权重到1维float型指针fc1w中,导入fc1偏置到1维float型指针fc1b中
	//const char* fc1w_filename = combine_strings(basedir, "f6w_t.bin");   //c1卷积权重路径
	//float* fc1w = read_weight(fc1w_filename, 7, 32, 128);      //卷积权重文件 、卷积核尺寸、卷积核个数   
	//printf("导入fc1卷积权重成功!\n");
	////导入fc1偏置
	//const char* fc1b_filename = combine_strings(basedir, "f6b.bin");   //c1偏置路径
	//float* fc1b = read_bias(fc1b_filename, 128);    //偏置文件、偏置元素个数    
	//printf("导入fc1偏置成功!\n");
	////求fc1卷积和激活
	////参数：该层的权重指针，输入图像数据指针、输入宽、输入高、输入通道数、卷积核的个数、卷积核的大小、步幅大小、补边否,激活函数类型
	////int pad=(fsize-1)/2;
	//clock_t t9, t10;
	//t9 = clock();

	//printf("开始fc1计算...... \n");
	//out = nn(fc1w, fc1b, out, 1568,128, 1);  //pad=0(vaild),Relu激活


	//t10 = clock();
	//printf("fc1计算完成！耗时:%20.10f ms! \n", (double)(t10 - t9) / CLOCKS_PER_SEC*1000.0);
	//printf("\n");
	////int o;
	////for (o = 0; o< 128; o++)          //遍历输出列
	////{
	////	printf("%f  ", out[o]);
	////}
	////printf("\n");

	///*
	//fc2(softmax激活)
	//*/
	////导入fc1卷积权重到1维float型指针fc1w中,导入fc1偏置到1维float型指针fc1b中
	//const char* fc2w_filename = combine_strings(basedir, "f8w_t.bin");   //c1卷积权重路径
	//float* fc2w = read_weight(fc2w_filename, 1, 128, 10);      //卷积权重文件 、卷积核尺寸、卷积核个数   
	//printf("导入fc2卷积权重成功!\n");
	////导入fc1偏置
	//const char* fc2b_filename = combine_strings(basedir, "f8b.bin");   //c1偏置路径
	//float* fc2b = read_bias(fc2b_filename, 128);    //偏置文件、偏置元素个数    
	//printf("导入fc2偏置成功!\n");
	////求fc1卷积和激活
	////参数：该层的权重指针，输入图像数据指针、输入宽、输入高、输入通道数、卷积核的个数、卷积核的大小、步幅大小、补边否,激活函数类型
	////int pad=(fsize-1)/2;
	//clock_t t11, t12;
	//t11 = clock();

	//printf("开始fc2计算...... \n");
	//out = nn(fc2w, fc2b, out, 128, 10, 0);  //pad=0(vaild),softmax激活
	//t12= clock();
	//printf("fc2计算完成！耗时:%20.10f ms! \n", (double)(t12 - t11) / CLOCKS_PER_SEC*1000.0);
	//printf("\n");

	//for (c = 0; c< 10; c++)          //遍历输出列
	//{
	//	printf("%f  ", out[c]);
	//}
	//printf("\n");


	/*用卷积实现全连接计算
	fc1(relu激活)
	*/
	//导入fc1卷积权重到1维float型指针fc1w中,导入fc1偏置到1维float型指针fc1b中
	const char* fc1w_filename = combine_strings(basedir, "f6w_t.bin");   //c1卷积权重路径
	float* fc1w = read_weight(fc1w_filename, 7, 32, 128);      //卷积权重文件 、卷积核尺寸、卷积核个数   
	printf("导入fc1卷积权重成功!\n");
	//导入fc1偏置
	const char* fc1b_filename = combine_strings(basedir, "f6b.bin");   //c1偏置路径
	float* fc1b = read_bias(fc1b_filename, 128);    //偏置文件、偏置元素个数    
	printf("导入fc1偏置成功!\n");
	//求fc1卷积和激活
	//参数：该层的权重指针，输入图像数据指针、输入宽、输入高、输入通道数、卷积核的个数、卷积核的大小、步幅大小、补边否,激活函数类型
	//int pad=(fsize-1)/2;
	clock_t t9, t10;
	t9 = clock();
	printf("开始fc1计算...... \n");
	out = conv(fc1w, fc1b, out, 7, 7, 32, 128, 7, 1, 0, 1);  //pad=0(vaild),Relu激活
	t10 = clock();
	printf("fc1计算完成！耗时:%20.10f ms! \n", (double)(t10 - t9) / CLOCKS_PER_SEC*1000.0);
	printf("\n");
	//int o;
	//for (o = 0; o< 128; o++)          //遍历输出列
	//{
	//	printf("%f  ", out[o]);
	//}
	//printf("\n");


	/*
	fc2(softmax激活)
	*/
	//导入fc2卷积权重到1维float型指针c1w中,导入fc2偏置到1维float型指针fc2b中
	const char* fc2w_filename = combine_strings(basedir, "f8w_t.bin");   //fc2卷积权重路径
	float* fc2w = read_weight(fc2w_filename, 1, 128, 10);      //卷积权重文件 、卷积核尺寸、卷积核个数   
	printf("导入fc12卷积权重成功!\n");
	//导入fc2偏置
	const char* fc2b_filename = combine_strings(basedir, "f8b.bin");   //c1偏置路径
	float* fc2b = read_bias(fc2b_filename, 10);    //偏置文件、偏置元素个数    
	printf("导入fc12偏置成功!\n");
	//求fc2卷积和激活
	//参数：该层的权重指针，输入图像数据指针、输入宽、输入高、输入通道数、卷积核的个数、卷积核的大小、步幅大小、补边否,激活类型
	//int pad=(fsize-1)/2;
	clock_t t11, t12;
	t11= clock();
	printf("开始fc2计算...... \n");
	out = conv(fc2w, fc2b, out, 1, 1, 128, 10, 1, 1, 0, 0);  //pad=0(vaild),softmax激活
	t12 = clock();
	printf("fc2计算完成！耗时:%20.10f ms! \n", (double)(t12 - t11) / CLOCKS_PER_SEC*1000.0);
	printf("\n");


	/*
	验证最后的结果
	*/
	int index=0;
	float max = 0.0;
	for (p = 0; p< 10; p++)          //遍历输出列
	{
		printf("%f  ", out[p]);
		if (max <= out[p])
		{
			index = p;
			max = out[p];
		}
	}
	printf("\n");
	printf("最终预测结果：%d",index);
	system("pause");

	
}