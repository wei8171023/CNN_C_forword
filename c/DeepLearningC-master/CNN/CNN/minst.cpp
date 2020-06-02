#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "minst.h"

//float32的二进制数据在计算机中从低位到高位存储
//英特尔处理器和其他低端机用户必须翻转头字节。  
int ReverseInt(int i)   
{  
	unsigned char ch1, ch2, ch3, ch4;  
	ch1 = i & 255;              //取实际的最高8位
	ch2 = (i >> 8) & 255;      //取实际的次高8位
	ch3 = (i >> 16) & 255;     //取实际的次低8位
	ch4 = (i >> 24) & 255;     //取实际的最低8位
	return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4; 
	//return ((short)ch1 << 8) + ch2;
}


// 读入图像像素值(2维float型)
MinstImg read_Img(const char* filename) 
{
	FILE  *fp=NULL;                     //定义一个指向文件的指针变量fp
	fp=fopen(filename,"rb");            //以二进制格式打开图像文件
	if(fp==NULL)
		printf("open file failed\n");
	assert(fp);
	printf("---读取图片文件成功！ \n");
	 
	//int number_of_images = 1;      //输入图片的个数==1

	int buf_w[1];         //定义接收图像宽数组
	int buf_h[1];         //定义接收图像高数组
	int buf_ch[1];        //定义接收图像通道数数组
	//获取图像的宽度 
	fread(buf_w,sizeof(int),1,fp); 
	//获取图像的高度  
	fread(buf_h, sizeof(int), 1, fp);
	//获取图像的通道数  
	fread(buf_ch, sizeof(int), 1, fp);
	printf("---输入图像宽：%d,输入图像高：%d,通道数：%d \n", buf_w[0], buf_h[0],buf_ch[0]);

	// 图像数组的初始化，将图片的宽、高、通道数、像素数据保存到图像结构体minstimg中。
	MinstImg minstimg = {0};     //新建Minsting图像结构体
	minstimg.h = buf_h[0];       //图像的高
	minstimg.w = buf_w[0];       //图像的宽
	minstimg.ch = buf_ch[0];     //图像的通道数
	minstimg.ImgData = (float* )calloc(buf_h[0] * buf_w[0] * buf_ch[0] , sizeof(float));    //为图片数组分配一维内存空间
	
	int rc1, i = 0;       //定义图像的像素个数i
	float buf[1];           //定义接收的像素值数组
	while ((rc1 = fread(buf, sizeof(float), 1, fp)) != 0)  //循环读取源文件中的一个float数据，复制到buf
	{
		/*printf("%f--",buf[0]);*/

		float pixel = buf[0]/ 255.0;
		//printf("%f  ", buf[0]);
		//printf("%f  ", pixel);
		minstimg.ImgData[i++] = pixel;                   //将复制的buf中的数据，拷贝到权重指针weigth中
	}
	printf("---图像的像素个数：%d \n", i);
	
	fclose(fp);
	return minstimg;

}
// 读入权重到float类型的指针
float* read_weight(const char* filename, int fsize,int inchan, int outchan)
{
	FILE  *fp = NULL;                     //定义一个指向文件的指针变量fp
	fp = fopen(filename, "rb");            //以二进制格式打开权重文件
	if (fp == NULL)
		printf("open file failed\n");
	assert(fp);
	printf("---读取权重文件成功！ \n");
	

	float* weight = (float*)calloc(fsize*fsize*inchan*outchan,sizeof(float));      //定义获取的权重指针
	int rc1, i = 0;              //定义权重个数
	float buf[1];
	while ((rc1 = fread(buf, sizeof(float), 1, fp)) != 0)    //循环读取源文件中的一个float数据，复制到buf
	{
		/*printf("%f--",buf[0]);*/
		//printf("%f  ", buf[0]);
		weight[i++] = buf[0];                //将复制的buf中的数据，拷贝到权重指针weigth中
	}	
	printf("---权重的个数：%d  \n",i);

	fclose(fp);
	return weight;
}

// 读入权重到float类型的指针
float* read_bias(const char* filename, int num_filter)
{
	FILE  *fp = NULL;                     //定义一个指向文件的指针变量fp
	fp = fopen(filename, "rb");            //以二进制格式打开权重文件
	if (fp == NULL)
		printf("open file failed\n");
	assert(fp);
	printf("---读取偏置文件成功！\n");

	float* bias = (float*)calloc(num_filter,sizeof(float));      //定义获取的权重指针
	int rc1, i = 0;              //定义权重个数
	float buf[1];
	while ((rc1 = fread(buf, sizeof(float), 1, fp)) != 0)    //循环读取源文件中的一个float数据，复制到buf
	{
		/*printf("%f--",buf[0]);*/
		//printf("%f  ", buf[0]);
		bias[i++] = buf[0];                //将复制的buf中的数据，拷贝到权重指针weigth中
	}
	printf("---偏置的个数：%d   \n", i);
	fclose(fp);
	return bias;
}
//LabelArr read_Lable(const char* filename)          // 读入图像的标签
//{
//	FILE  *fp=NULL;
//	fp=fopen(filename,"rb");               //以二进制格式打开图像标签文件
//	if(fp==NULL)
//		printf("open file failed\n");     
//	assert(fp);
//
//	int magic_number = 0;                 
//	int number_of_labels = 0;			//标签的数量
//	int label_long = 10;                //标签的长度
//
//	//从文件中读取sizeof(magic_number) 个字符到 ---magic_number  
//	fread((char*)&magic_number,sizeof(magic_number),1,fp); 
//	magic_number = ReverseInt(magic_number);			 //数值翻转提取
//
//	//获取训练或测试image的个数---number_of_images 
//	fread((char*)&number_of_labels,sizeof(number_of_labels),1,fp);  
//	number_of_labels = ReverseInt(number_of_labels);    //数值翻转提取
//
//	int i,l;
//
//	// 图像标签数组的初始化
//	LabelArr labarr=(LabelArr)malloc(sizeof(MinstLabelArr));
//	labarr->LabelNum=number_of_labels;											  //标签的数目                  
//	labarr->LabelPtr=(MinstLabel*)malloc(number_of_labels*sizeof(MinstLabel));    //标签数组分配动态内存
//
//	for(i = 0; i < number_of_labels; ++i)    //遍历标签数目
//	{  
//		// 数据库内的图像标签是一位，这里将图像标签变成10位，10位中只有唯一一位为1，为1位即是图像标记
//		labarr->LabelPtr[i].l=10;												//第i个标签维度为10
//		labarr->LabelPtr[i].LabelData=(float*)calloc(label_long,sizeof(float)); //为标签值分配动态内存
//		unsigned char temp = 0;  
//		fread((char*) &temp, sizeof(temp),1,fp);								//获取一个图像的便签
//		labarr->LabelPtr[i].LabelData[(int)temp]=1.0;							//初始化便签---给temp数组赋值1
//	}
//
//	fclose(fp);
//	return labarr;	
//}

char* intTochar(int i)						// 将整数型数字i转换成字符串型数字ptr
{
	int itemp=i;                  
	int w=0;                      
	while(itemp>=10){      
		itemp=itemp/10;    //
		w++;               //数字i的位数-1，10的倍数
	}
	//为输出的字符分配空间（+2表示多一位，一个结尾存储'\0',）
	char* ptr=(char*)malloc((w+2)*sizeof(char));   
	ptr[w+1]='\0';              //输出字符串结尾赋值为空字符
	int r; // 余数
	while(i>=10){              //整数型数字i转为字符串型数组ptr
		r=i%10;           //取余数，即最低位数字
		i=i/10;		      //取商，即高几位数字
		ptr[w]=(char)(r+48);  //将整数型r转为字符型，（字符型0的ASCII码值为48，因此要加48）
		w--;                  
	}
	ptr[w]=(char)(i+48);       //将数字i高位赋值给字符串的最高位
	return ptr;
}

// 将两个字符串相连,函数返回指针值
char * combine_strings(char *a, char *b) 
{
	char *ptr;              ///拼接后的字符串
	int lena=strlen(a),lenb=strlen(b);     //字符串a,b的长度
	int i,l=0;   
	ptr = (char *)malloc((lena+lenb+1) * sizeof(char));    //为拼接后的字符串分配内存空间
	for(i=0;i<lena;i++)       
		ptr[l++]=a[i];
	for(i=0;i<lenb;i++)
		ptr[l++]=b[i];
	ptr[l]='\0';           //拼接后的数组赋值为'\0'
	return (ptr);     //返回拼接后字符串指针
}

//void save_Img(ImgArr imgarr, char* filedir)  // 将图像数结构体保存成x.gray文件
//{
//	int img_number = imgarr->ImgNum;         //获取图像结构体中的图像数目
//
//	int i, r;         //整型i,r
//	for (i = 0; i < img_number; i++){       //  遍历每一张图像
//		// 将输出路径、第几张图片、'.gray'拼接为输出文件名
//		const char* filename = combine_strings(filedir, combine_strings(intTochar(i), ".gray"));
//		FILE  *fp = NULL;
//		fp = fopen(filename, "wb");      //新建文件
//		if (fp == NULL)
//			printf("write file failed\n");
//		assert(fp);
//
//		for (r = 0; r < imgarr->ImgPtr[i].r; r++)  //一张图片按行进行复制到fp中
//			fwrite(imgarr->ImgPtr[i].ImgData[r], sizeof(float), imgarr->ImgPtr[i].c, fp);
//
//		fclose(fp);
//	}
//}