#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "minst.h"

//float32�Ķ����������ڼ�����дӵ�λ����λ�洢
//Ӣ�ض��������������Ͷ˻��û����뷭תͷ�ֽڡ�  
int ReverseInt(int i)   
{  
	unsigned char ch1, ch2, ch3, ch4;  
	ch1 = i & 255;              //ȡʵ�ʵ����8λ
	ch2 = (i >> 8) & 255;      //ȡʵ�ʵĴθ�8λ
	ch3 = (i >> 16) & 255;     //ȡʵ�ʵĴε�8λ
	ch4 = (i >> 24) & 255;     //ȡʵ�ʵ����8λ
	return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4; 
	//return ((short)ch1 << 8) + ch2;
}


// ����ͼ������ֵ(2άfloat��)
MinstImg read_Img(const char* filename) 
{
	FILE  *fp=NULL;                     //����һ��ָ���ļ���ָ�����fp
	fp=fopen(filename,"rb");            //�Զ����Ƹ�ʽ��ͼ���ļ�
	if(fp==NULL)
		printf("open file failed\n");
	assert(fp);
	printf("---��ȡͼƬ�ļ��ɹ��� \n");
	 
	//int number_of_images = 1;      //����ͼƬ�ĸ���==1

	int buf_w[1];         //�������ͼ�������
	int buf_h[1];         //�������ͼ�������
	int buf_ch[1];        //�������ͼ��ͨ��������
	//��ȡͼ��Ŀ�� 
	fread(buf_w,sizeof(int),1,fp); 
	//��ȡͼ��ĸ߶�  
	fread(buf_h, sizeof(int), 1, fp);
	//��ȡͼ���ͨ����  
	fread(buf_ch, sizeof(int), 1, fp);
	printf("---����ͼ���%d,����ͼ��ߣ�%d,ͨ������%d \n", buf_w[0], buf_h[0],buf_ch[0]);

	// ͼ������ĳ�ʼ������ͼƬ�Ŀ��ߡ�ͨ�������������ݱ��浽ͼ��ṹ��minstimg�С�
	MinstImg minstimg = {0};     //�½�Minstingͼ��ṹ��
	minstimg.h = buf_h[0];       //ͼ��ĸ�
	minstimg.w = buf_w[0];       //ͼ��Ŀ�
	minstimg.ch = buf_ch[0];     //ͼ���ͨ����
	minstimg.ImgData = (float* )calloc(buf_h[0] * buf_w[0] * buf_ch[0] , sizeof(float));    //ΪͼƬ�������һά�ڴ�ռ�
	
	int rc1, i = 0;       //����ͼ������ظ���i
	float buf[1];           //������յ�����ֵ����
	while ((rc1 = fread(buf, sizeof(float), 1, fp)) != 0)  //ѭ����ȡԴ�ļ��е�һ��float���ݣ����Ƶ�buf
	{
		/*printf("%f--",buf[0]);*/

		float pixel = buf[0]/ 255.0;
		//printf("%f  ", buf[0]);
		//printf("%f  ", pixel);
		minstimg.ImgData[i++] = pixel;                   //�����Ƶ�buf�е����ݣ�������Ȩ��ָ��weigth��
	}
	printf("---ͼ������ظ�����%d \n", i);
	
	fclose(fp);
	return minstimg;

}
// ����Ȩ�ص�float���͵�ָ��
float* read_weight(const char* filename, int fsize,int inchan, int outchan)
{
	FILE  *fp = NULL;                     //����һ��ָ���ļ���ָ�����fp
	fp = fopen(filename, "rb");            //�Զ����Ƹ�ʽ��Ȩ���ļ�
	if (fp == NULL)
		printf("open file failed\n");
	assert(fp);
	printf("---��ȡȨ���ļ��ɹ��� \n");
	

	float* weight = (float*)calloc(fsize*fsize*inchan*outchan,sizeof(float));      //�����ȡ��Ȩ��ָ��
	int rc1, i = 0;              //����Ȩ�ظ���
	float buf[1];
	while ((rc1 = fread(buf, sizeof(float), 1, fp)) != 0)    //ѭ����ȡԴ�ļ��е�һ��float���ݣ����Ƶ�buf
	{
		/*printf("%f--",buf[0]);*/
		//printf("%f  ", buf[0]);
		weight[i++] = buf[0];                //�����Ƶ�buf�е����ݣ�������Ȩ��ָ��weigth��
	}	
	printf("---Ȩ�صĸ�����%d  \n",i);

	fclose(fp);
	return weight;
}

// ����Ȩ�ص�float���͵�ָ��
float* read_bias(const char* filename, int num_filter)
{
	FILE  *fp = NULL;                     //����һ��ָ���ļ���ָ�����fp
	fp = fopen(filename, "rb");            //�Զ����Ƹ�ʽ��Ȩ���ļ�
	if (fp == NULL)
		printf("open file failed\n");
	assert(fp);
	printf("---��ȡƫ���ļ��ɹ���\n");

	float* bias = (float*)calloc(num_filter,sizeof(float));      //�����ȡ��Ȩ��ָ��
	int rc1, i = 0;              //����Ȩ�ظ���
	float buf[1];
	while ((rc1 = fread(buf, sizeof(float), 1, fp)) != 0)    //ѭ����ȡԴ�ļ��е�һ��float���ݣ����Ƶ�buf
	{
		/*printf("%f--",buf[0]);*/
		//printf("%f  ", buf[0]);
		bias[i++] = buf[0];                //�����Ƶ�buf�е����ݣ�������Ȩ��ָ��weigth��
	}
	printf("---ƫ�õĸ�����%d   \n", i);
	fclose(fp);
	return bias;
}
//LabelArr read_Lable(const char* filename)          // ����ͼ��ı�ǩ
//{
//	FILE  *fp=NULL;
//	fp=fopen(filename,"rb");               //�Զ����Ƹ�ʽ��ͼ���ǩ�ļ�
//	if(fp==NULL)
//		printf("open file failed\n");     
//	assert(fp);
//
//	int magic_number = 0;                 
//	int number_of_labels = 0;			//��ǩ������
//	int label_long = 10;                //��ǩ�ĳ���
//
//	//���ļ��ж�ȡsizeof(magic_number) ���ַ��� ---magic_number  
//	fread((char*)&magic_number,sizeof(magic_number),1,fp); 
//	magic_number = ReverseInt(magic_number);			 //��ֵ��ת��ȡ
//
//	//��ȡѵ�������image�ĸ���---number_of_images 
//	fread((char*)&number_of_labels,sizeof(number_of_labels),1,fp);  
//	number_of_labels = ReverseInt(number_of_labels);    //��ֵ��ת��ȡ
//
//	int i,l;
//
//	// ͼ���ǩ����ĳ�ʼ��
//	LabelArr labarr=(LabelArr)malloc(sizeof(MinstLabelArr));
//	labarr->LabelNum=number_of_labels;											  //��ǩ����Ŀ                  
//	labarr->LabelPtr=(MinstLabel*)malloc(number_of_labels*sizeof(MinstLabel));    //��ǩ������䶯̬�ڴ�
//
//	for(i = 0; i < number_of_labels; ++i)    //������ǩ��Ŀ
//	{  
//		// ���ݿ��ڵ�ͼ���ǩ��һλ�����ｫͼ���ǩ���10λ��10λ��ֻ��ΨһһλΪ1��Ϊ1λ����ͼ����
//		labarr->LabelPtr[i].l=10;												//��i����ǩά��Ϊ10
//		labarr->LabelPtr[i].LabelData=(float*)calloc(label_long,sizeof(float)); //Ϊ��ǩֵ���䶯̬�ڴ�
//		unsigned char temp = 0;  
//		fread((char*) &temp, sizeof(temp),1,fp);								//��ȡһ��ͼ��ı�ǩ
//		labarr->LabelPtr[i].LabelData[(int)temp]=1.0;							//��ʼ����ǩ---��temp���鸳ֵ1
//	}
//
//	fclose(fp);
//	return labarr;	
//}

char* intTochar(int i)						// ������������iת�����ַ���������ptr
{
	int itemp=i;                  
	int w=0;                      
	while(itemp>=10){      
		itemp=itemp/10;    //
		w++;               //����i��λ��-1��10�ı���
	}
	//Ϊ������ַ�����ռ䣨+2��ʾ��һλ��һ����β�洢'\0',��
	char* ptr=(char*)malloc((w+2)*sizeof(char));   
	ptr[w+1]='\0';              //����ַ�����β��ֵΪ���ַ�
	int r; // ����
	while(i>=10){              //����������iתΪ�ַ���������ptr
		r=i%10;           //ȡ�����������λ����
		i=i/10;		      //ȡ�̣����߼�λ����
		ptr[w]=(char)(r+48);  //��������rתΪ�ַ��ͣ����ַ���0��ASCII��ֵΪ48�����Ҫ��48��
		w--;                  
	}
	ptr[w]=(char)(i+48);       //������i��λ��ֵ���ַ��������λ
	return ptr;
}

// �������ַ�������,��������ָ��ֵ
char * combine_strings(char *a, char *b) 
{
	char *ptr;              ///ƴ�Ӻ���ַ���
	int lena=strlen(a),lenb=strlen(b);     //�ַ���a,b�ĳ���
	int i,l=0;   
	ptr = (char *)malloc((lena+lenb+1) * sizeof(char));    //Ϊƴ�Ӻ���ַ��������ڴ�ռ�
	for(i=0;i<lena;i++)       
		ptr[l++]=a[i];
	for(i=0;i<lenb;i++)
		ptr[l++]=b[i];
	ptr[l]='\0';           //ƴ�Ӻ�����鸳ֵΪ'\0'
	return (ptr);     //����ƴ�Ӻ��ַ���ָ��
}

//void save_Img(ImgArr imgarr, char* filedir)  // ��ͼ�����ṹ�屣���x.gray�ļ�
//{
//	int img_number = imgarr->ImgNum;         //��ȡͼ��ṹ���е�ͼ����Ŀ
//
//	int i, r;         //����i,r
//	for (i = 0; i < img_number; i++){       //  ����ÿһ��ͼ��
//		// �����·�����ڼ���ͼƬ��'.gray'ƴ��Ϊ����ļ���
//		const char* filename = combine_strings(filedir, combine_strings(intTochar(i), ".gray"));
//		FILE  *fp = NULL;
//		fp = fopen(filename, "wb");      //�½��ļ�
//		if (fp == NULL)
//			printf("write file failed\n");
//		assert(fp);
//
//		for (r = 0; r < imgarr->ImgPtr[i].r; r++)  //һ��ͼƬ���н��и��Ƶ�fp��
//			fwrite(imgarr->ImgPtr[i].ImgData[r], sizeof(float), imgarr->ImgPtr[i].c, fp);
//
//		fclose(fp);
//	}
//}