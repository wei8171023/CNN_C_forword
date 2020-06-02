#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"
#include "minst.h"


/*������*/
int main()
{	
	//�������ͼƬ��ͼ��ṹ��Img
	printf("��ʼ����ͼ��.....\n");
	MinstImg Img = read_Img("D:\\FPGA-AI\\Lenet_MNIST\\DeepLearningC-master\\CNN\\CNNdata\\input_struct11_float32_whc_6_8.bin"); 
	printf("����ͼ��ɹ�!\n");
	printf("\n");
	//int l, m, n;
	//for (l= 0; l < 1; l++)    //�������ͨ��==����˸���
	//	for (m = 0; m < 28; m++)         //���������
	//		for (n= 0; n< 28; n++)          //���������
	//			printf("%f ", Img.ImgData[l * 28 * 28 + m * 28 + n]);
	//	printf("\n");
	

	//Ȩ�صĻ���ַ
	char* basedir = "D:\\FPGA-AI\\Lenet_MNIST\\DeepLearningC-master\\CNN\\weight_bin\\";
	//�������������ͼ
	float* out;
	/*
	conv1(rule����)�������������ͼout
	*/
	//������ͼƬ�л�ȡ����Ŀ���
	int imgw = Img.h;           
	int imgh = Img.w;            
	int imgch = Img.ch;      
	//����c1Ȩ��
	//����c1���Ȩ�ص�1άfloat��ָ��c1w��,����c1ƫ�õ�1άfloat��ָ��c1b��
	printf("��ʼ����c1���Ȩ��.....\n");
	const char* c1w_filename = combine_strings(basedir, "c1w.bin");   //c1���Ȩ��·��
	float* c1w = read_weight(c1w_filename,3,1,16);      //���Ȩ���ļ� ������˳ߴ硢����˸���   
	printf("����c1���Ȩ�سɹ�!\n");
	//����c1ƫ��
	printf("��ʼ����c1ƫ��.....\n");
	const char* c1b_filename = combine_strings(basedir, "c1b.bin");   //c1ƫ��·��
	float* c1b = read_bias(c1b_filename,16);            //ƫ���ļ���ƫ��Ԫ�ظ���    
	printf("����c1ƫ�óɹ�!\n");
	//��ʱ��ʼ
	clock_t t1, t2;
	t1 = clock();
	printf("��ʼconv1����..... \n");
	//��conv1
	//�������ò��Ȩ��ָ�룬����ͼ������ָ�롢���������ߡ�����ͨ���������ͨ����������˵Ĵ�С��������С�����߷�	
	out = conv(c1w,c1b,Img.ImgData,28,28,1,16,3,1,1,1);
	//��ʱ����
	t2 = clock();
	
	printf("conv1�������,��ʱ:%20.10f ms! \n", (double)(t2 - t1) / CLOCKS_PER_SEC*1000.0);
	printf("\n");

	/*
	maxpool1,�����������ͼout
	*/
	clock_t t3, t4;
	t3 = clock();
	printf("��ʼpool����.....  \n");
	out = maxpool(out, 28, 28, 16, 2, 2);
	t4 = clock();
	printf("pool1�������,��ʱ:%20.10f ms! \n", (double)(t4 - t3) / CLOCKS_PER_SEC*1000.0);
	printf("\n");

	/*
	conv2(relu����),�����������ͼout
	*/
	//����c2���Ȩ�ص�1άfloat��ָ��c2w��,����c2ƫ�õ�1άfloat��ָ��c2b��
	const char* c2w_filename = combine_strings(basedir, "c2w.bin");   //c1���Ȩ��·��
	float* c2w = read_weight(c2w_filename, 3, 16, 32);      //���Ȩ���ļ� ������˳ߴ硢����˸���   
	printf("����c2���Ȩ�سɹ�!\n");
	//����c2ƫ��
	const char* c2b_filename = combine_strings(basedir, "c2b.bin");   //c1ƫ��·��
	float* c2b = read_bias(c2b_filename, 32);    //ƫ���ļ���ƫ��Ԫ�ظ���    
	printf("����c2ƫ�óɹ�!\n");
	//��c2����ͼ���
	//�������ò��Ȩ��ָ�룬����ͼ������ָ�롢���������ߡ�����ͨ����������˵ĸ���������˵Ĵ�С��������С�����߷�
	//int pad=(fsize-1)/2;
	clock_t t5, t6;
	t5 = clock();
	printf("��ʼconv2����...... \n");
	out = conv(c2w, c2b, out, 14, 14, 16, 32, 3, 1, 1,1);
	t6 = clock();
	printf("conv2������ɣ���ʱ:%20.10f ms! \n", (double)(t6 - t5) / CLOCKS_PER_SEC*1000.0);
	printf("\n");
	/*
	maxpool2,�����������ͼout
	*/
	clock_t t7, t8;
	t7 = clock();
	printf("��ʼpoo2����.....  \n");
	out = maxpool(out, 14, 14, 32, 2, 2);
	t8 = clock();
	printf("pool2�������,��ʱ:%20.10f ms! \n", (double)(t8 - t7) / CLOCKS_PER_SEC*1000.0);
	printf("\n");

	printf("����pool2�ļ�������  \n");
	int i, j, k;
	//for (k = 0; k < 32; k++)    //�������ͨ��==����˸���
	//{
	//	for (i = 0; i < 7; i++)         //���������
	//	{
	//		for (j = 0; j < 7; j++)          //���������
	//		{
	//			printf("%f  ", out[k * 7 * 7 + i * 7 + j]);
	//		}
	//	}
	//	printf("\n");
	//}

	/*
	flatten   ���ػ�����Ľ����32,7,7��cfirst����˳��ת��Ϊ��7,7��32��clastģʽ������Ӧkeras��Ȩ�غ����ݵ�clast˳��
	*/
	printf("��ʼflatten \n");
	float** v = (float**)malloc(32 * sizeof(float*));      //Ϊת��ǰ�Ķ�άָ��v�����ڴ�ռ�[32][7*7]
	for (i = (32- 1); i != (-1); i--)
	{
		v[i] = (float*)calloc(49, sizeof(float));
	}
	//float* y = (float*)calloc(1568, sizeof(float));

	float** y = (float**)malloc(49 * sizeof(float*));      //Ϊת�ú�Ķ�άָ��y�����ڴ�ռ�[7*7][32]
	for (i = (49 - 1); i != (-1); i--)
	{
		y[i] = (float*)calloc(32, sizeof(float));
	}
	//float* y = (float*)calloc(1568, sizeof(float));

	int r, c;
	int l = 0;
	for (j = 0; j < 32; j++)//16     //3ά(1ά)ת2ά
	{

		for (r = 0; r < 7; r++)
		{
			for (c = 0; c < 7; c++)
			{
				v[j][l] = out[j*49+r*7+c];
				l++;
			}
		}
		l = 0;//��ʼ��һ�и���
	}
	printf("һάת��ά��ɣ� \n");

	int p, d;
	float temp = 0.0;
	for (i = 0; i < 32; i++)         //2ά����vת�ã���2ά����y
	{
		for (int j = 0; j < 49; j++)
		{
			y[j][i] = v[i][j];
		}
	}
	printf("��άvת����ɣ� \n");

	k = 0;
	for (int p = 0; p < 49; p++)      //2άyת��1άout��
	{
		for (int d = 0; d < 32; d++)
		{

			out[k] = y[p][d];
			k++;
		}
	}
	printf("��άvתһάout��ɣ� \n");

	//printf("flatten��Ľ����  \n");
	//for (k = 0; k < 32; k++)    //�������ͨ��==����˸���
	//{
	//	for (i = 0; i < 7; i++)         //���������
	//	{
	//		for (j = 0; j < 7; j++)          //���������
	//		{
	//			printf("%f  ", out[k * 7 * 7 + i * 7 + j]);
	//		}
	//	}
	//	printf("\n");
	//}


	///*
	//fc1(relu����)
	//*/
	////����fc1���Ȩ�ص�1άfloat��ָ��fc1w��,����fc1ƫ�õ�1άfloat��ָ��fc1b��
	//const char* fc1w_filename = combine_strings(basedir, "f6w_t.bin");   //c1���Ȩ��·��
	//float* fc1w = read_weight(fc1w_filename, 7, 32, 128);      //���Ȩ���ļ� ������˳ߴ硢����˸���   
	//printf("����fc1���Ȩ�سɹ�!\n");
	////����fc1ƫ��
	//const char* fc1b_filename = combine_strings(basedir, "f6b.bin");   //c1ƫ��·��
	//float* fc1b = read_bias(fc1b_filename, 128);    //ƫ���ļ���ƫ��Ԫ�ظ���    
	//printf("����fc1ƫ�óɹ�!\n");
	////��fc1����ͼ���
	////�������ò��Ȩ��ָ�룬����ͼ������ָ�롢���������ߡ�����ͨ����������˵ĸ���������˵Ĵ�С��������С�����߷�,���������
	////int pad=(fsize-1)/2;
	//clock_t t9, t10;
	//t9 = clock();

	//printf("��ʼfc1����...... \n");
	//out = nn(fc1w, fc1b, out, 1568,128, 1);  //pad=0(vaild),Relu����


	//t10 = clock();
	//printf("fc1������ɣ���ʱ:%20.10f ms! \n", (double)(t10 - t9) / CLOCKS_PER_SEC*1000.0);
	//printf("\n");
	////int o;
	////for (o = 0; o< 128; o++)          //���������
	////{
	////	printf("%f  ", out[o]);
	////}
	////printf("\n");

	///*
	//fc2(softmax����)
	//*/
	////����fc1���Ȩ�ص�1άfloat��ָ��fc1w��,����fc1ƫ�õ�1άfloat��ָ��fc1b��
	//const char* fc2w_filename = combine_strings(basedir, "f8w_t.bin");   //c1���Ȩ��·��
	//float* fc2w = read_weight(fc2w_filename, 1, 128, 10);      //���Ȩ���ļ� ������˳ߴ硢����˸���   
	//printf("����fc2���Ȩ�سɹ�!\n");
	////����fc1ƫ��
	//const char* fc2b_filename = combine_strings(basedir, "f8b.bin");   //c1ƫ��·��
	//float* fc2b = read_bias(fc2b_filename, 128);    //ƫ���ļ���ƫ��Ԫ�ظ���    
	//printf("����fc2ƫ�óɹ�!\n");
	////��fc1����ͼ���
	////�������ò��Ȩ��ָ�룬����ͼ������ָ�롢���������ߡ�����ͨ����������˵ĸ���������˵Ĵ�С��������С�����߷�,���������
	////int pad=(fsize-1)/2;
	//clock_t t11, t12;
	//t11 = clock();

	//printf("��ʼfc2����...... \n");
	//out = nn(fc2w, fc2b, out, 128, 10, 0);  //pad=0(vaild),softmax����
	//t12= clock();
	//printf("fc2������ɣ���ʱ:%20.10f ms! \n", (double)(t12 - t11) / CLOCKS_PER_SEC*1000.0);
	//printf("\n");

	//for (c = 0; c< 10; c++)          //���������
	//{
	//	printf("%f  ", out[c]);
	//}
	//printf("\n");


	/*�þ��ʵ��ȫ���Ӽ���
	fc1(relu����)
	*/
	//����fc1���Ȩ�ص�1άfloat��ָ��fc1w��,����fc1ƫ�õ�1άfloat��ָ��fc1b��
	const char* fc1w_filename = combine_strings(basedir, "f6w_t.bin");   //c1���Ȩ��·��
	float* fc1w = read_weight(fc1w_filename, 7, 32, 128);      //���Ȩ���ļ� ������˳ߴ硢����˸���   
	printf("����fc1���Ȩ�سɹ�!\n");
	//����fc1ƫ��
	const char* fc1b_filename = combine_strings(basedir, "f6b.bin");   //c1ƫ��·��
	float* fc1b = read_bias(fc1b_filename, 128);    //ƫ���ļ���ƫ��Ԫ�ظ���    
	printf("����fc1ƫ�óɹ�!\n");
	//��fc1����ͼ���
	//�������ò��Ȩ��ָ�룬����ͼ������ָ�롢���������ߡ�����ͨ����������˵ĸ���������˵Ĵ�С��������С�����߷�,���������
	//int pad=(fsize-1)/2;
	clock_t t9, t10;
	t9 = clock();
	printf("��ʼfc1����...... \n");
	out = conv(fc1w, fc1b, out, 7, 7, 32, 128, 7, 1, 0, 1);  //pad=0(vaild),Relu����
	t10 = clock();
	printf("fc1������ɣ���ʱ:%20.10f ms! \n", (double)(t10 - t9) / CLOCKS_PER_SEC*1000.0);
	printf("\n");
	//int o;
	//for (o = 0; o< 128; o++)          //���������
	//{
	//	printf("%f  ", out[o]);
	//}
	//printf("\n");


	/*
	fc2(softmax����)
	*/
	//����fc2���Ȩ�ص�1άfloat��ָ��c1w��,����fc2ƫ�õ�1άfloat��ָ��fc2b��
	const char* fc2w_filename = combine_strings(basedir, "f8w_t.bin");   //fc2���Ȩ��·��
	float* fc2w = read_weight(fc2w_filename, 1, 128, 10);      //���Ȩ���ļ� ������˳ߴ硢����˸���   
	printf("����fc12���Ȩ�سɹ�!\n");
	//����fc2ƫ��
	const char* fc2b_filename = combine_strings(basedir, "f8b.bin");   //c1ƫ��·��
	float* fc2b = read_bias(fc2b_filename, 10);    //ƫ���ļ���ƫ��Ԫ�ظ���    
	printf("����fc12ƫ�óɹ�!\n");
	//��fc2����ͼ���
	//�������ò��Ȩ��ָ�룬����ͼ������ָ�롢���������ߡ�����ͨ����������˵ĸ���������˵Ĵ�С��������С�����߷�,��������
	//int pad=(fsize-1)/2;
	clock_t t11, t12;
	t11= clock();
	printf("��ʼfc2����...... \n");
	out = conv(fc2w, fc2b, out, 1, 1, 128, 10, 1, 1, 0, 0);  //pad=0(vaild),softmax����
	t12 = clock();
	printf("fc2������ɣ���ʱ:%20.10f ms! \n", (double)(t12 - t11) / CLOCKS_PER_SEC*1000.0);
	printf("\n");


	/*
	��֤���Ľ��
	*/
	int index=0;
	float max = 0.0;
	for (p = 0; p< 10; p++)          //���������
	{
		printf("%f  ", out[p]);
		if (max <= out[p])
		{
			index = p;
			max = out[p];
		}
	}
	printf("\n");
	printf("����Ԥ������%d",index);
	system("pause");

	
}