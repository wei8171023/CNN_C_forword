#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include <windows.h>
#include "cnn.h"


//卷积运算，
//参数：该层的权重指针，偏置指针，输入图像数据指针、输入宽、输入高、输入通道数、输出通道数、卷积核的大小、步幅大小、补边否（1是进行补边,0是不补边）,激活函数（1是relu,0是softmax）
float* conv(float* weigth, float* bias, float* imgData, int inw, int inh, int inch, int outchan, int fsize, int stride, int pad, int activation)
{
	
	int outw = (inw + 2 * pad - fsize) / stride + 1;						//输出高
	int outh = (inh + 2 * pad - fsize) / stride + 1;						//输出宽
	printf("---卷积输出宽：%d，高：%d\n",outw,outh);

	
	//定义1维float型输出特征图指针，并未指针分配动态内存空间，该空间存储float型的数据;
	float* out = (float*)calloc(outw*outh*outchan, sizeof(float));

	int pad_w, pad_h;                      //定义padding后的图像宽、高
	//int filter_size=fsize*fsize*num_filter;					       //卷积总元素个数
	
	//根据Padding模式（1为SAME,0为vaild），计算padding后的图像宽、高
	pad_w = inw + pad*2*(fsize-1)/2;              //padding后的宽
	pad_h = inh + pad*2*(fsize - 1) / 2;          //padding后的高
	printf("padding输出宽：%d，高：%d\n", pad_w, pad_h);
	//system("pause");

	//定义1维float型padding型特征图指针，并为指针分配动态内存空间，该空间存储float型的数据
	float* img_pad = (float*)calloc(pad_h*pad_w*inch, sizeof(float));   
	if (pad == 1)
	{
		printf("---开始padding \n");
		int p, q,r,a=0;
		for (r = 0; r < inch; r++)    //遍历输入通道
		{
			for (p = 0; p < pad_h; p++)    //遍历pad行
			{
				for (q = 0; q < pad_w; q++)    //遍历pad宽
				{
					a++;
					img_pad[r*pad_h*pad_w + p*pad_w + q]=get_pixel(imgData,inh,inw,r,p,q,pad);
				}
			}
				
		}
		//printf("padding的像素总数：%d   ",a);   //900个
		printf("---padding 完成！\n");
	}
	else 
	{
		img_pad = imgData;
	}
  
	printf("---开始卷积计算 \n");
	int i, j,k,l,m,n,col=0,row=0;
	float sum = 0.0;
	#pragma omp parallel for
	for (k = 0; k < outchan; k++)    //遍历输出通道==卷积核个数
	{
		//printf("k==%d====================================================================\n ", k);
		#pragma omp parallel for 
		for (i = 0; i < outh; i++)         //遍历输出行
		{
			//printf("i==%d=====================================================\n ", i);
			#pragma omp parallel for 
			for (j = 0; j < outw; j++)          //遍历输出列
			{ 
				out[k*outh*outw + i*outw + j] = 0;
				//计算一次卷积操作和一次偏置相加操作
				//printf("j==%d===================================\n ", j);
				for (l = 0; l < inch; l++)            //遍历输入通道
				{
					//printf("l==%d======================\n ", l);
					#pragma omp parallel for
					for (m = 0; m < fsize; m++)         //遍历卷积核的高
					{
						//printf("m==%d=============\n ", m);
						#pragma omp parallel for
						for (n = 0; n < fsize; n++)       //遍历卷积核的宽
						{
							//printf("n==%d********\n ", n);
							row = stride*i + m;   //本次乘法在pad上的行=本次卷积在pad上的起始行+核内行偏移
							//printf("乘加行：%d ", row);
							col = stride*j + n;   //本次乘法在pad上的列=本次卷积在pad上的起始列+核内列偏移
							//printf("乘加列：%d ", col);
							out[k*outh*outw + i*outw + j] += weigth[k*fsize*fsize*inch + l*fsize*fsize + m*fsize + n] * img_pad[l*pad_h*pad_w + row*pad_w + col];
						}
					}
				}
				//printf("乘加结果：%f\n", out[k*outh*outw + i*outw + j]);
				//加偏置
				out[k*outh*outw + i*outw + j] += bias[k];   
				if (activation == 1)   //relu激活(卷积计算中)
				{
					if (out[k*outh*outw + i*outw + j] < 0 || out[k*outh*outw + i*outw + j] == 0)
					{
						out[k*outh*outw + i*outw + j] = 0;
					}
					else if (out[k*outh*outw + i*outw + j] > 0)
					{
						out[k*outh*outw + i*outw + j] = out[k*outh*outw + i*outw + j];
					}
										
				}
				else 
				{
					sum += exp(out[k* outh* outw + i* outw + j]);   //求softmax分母之和
				}
			}
		}
	}

	if (activation == 0)  //softmax激活（输出全连接层中）
	{
		int x, y, z;

		for (z = 0; z < outchan; z++)    //遍历输出通道==卷积核个数
		{
			for (x = 0; x < outh; x++)         //遍历输出行
			{
				for (y = 0; y < outw; y++)          //遍历输出列
				{
					out[z* outh * outw + x* outw + y] = exp(out[z* outh * outw + x* outw + y])/sum; //计算出每个输出的softmax值
				}
			}
		}
	}
	else
	{
		NULL;
	}
	
	printf("---卷积计算完成\n");
	return out;
}

//最大池化运算
//参数：         输入图像、输入高、输入宽、输入通道数、池化核大小，步幅大小
float* maxpool(float* imgData, int inh, int inw, int inchan, int psize, int stride)
{
	int outh, outw, outchan;
	outh = (inh - psize) / stride + 1;   //池化输出高
	outw = (inw - psize) / stride + 1;   //池化输出宽
	outchan = inchan;                    //输出通道数
	printf("---池化输出宽：%d，高：%d\n", outw, outh);
	//定义1维float型输出特征图指针，并为指针分配动态内存空间，该空间存储float型的数据;
	float* out = (float*)calloc(outw*outh*outchan, sizeof(float));

	int i, j, k,m,n,col,row;
	float max;
	#pragma omp parallel for
	for (k = 0; k < outchan; k++)     //遍历输出通道==卷积核个数
	{
		//printf("k==%d====================================================================\n ", k);
		#pragma omp parallel for
		for (i = 0; i < outh; i++)         //遍历输出行
		{
			//printf("i==%d=====================================================\n ", i);
			#pragma omp parallel for
			for (j = 0; j < outw; j++)          //遍历输出列
			{
				//printf("j==%d===================================\n ", j);
				//计算一次池化操作
				max = 0.0;
				//遍历该通道一个池化核，获取其中的最大值
				for (m = 0; m < psize; m++)   //遍历池化核的高
				{
					//printf("m==%d=============\n ", m);
					for (n = 0; n < psize; n++)    //遍历池化核的宽
					{
						//printf("n==%d********\n ", n);
						row = stride*i + m;   //本次池化在输入图像上的行=本次池化在输入图像上的起始行+核内偏移
						col = stride*j + n;   //本次池化在输入图像上的列=本次池化在输入图像上的起始列+核内偏移
						if (imgData[k*inh*inw + row*inh + col] > max)
						{
							max = imgData[k*inh*inw + row*inh + col];
						}
					}
				}
				out[k*outh*outw + i*outw + j] = max;
				//printf("%f  \n", out[k*outh*outw + i*outw + j]);
			}
		}
	}
	printf("---最大池化计算完成\n");
	return out;
}

//全连接运算
//参数： 该层的权重指针，偏置指针，输入图像数据指针，权重行数，       权重列数，     激活类型
float* nn(float* weigth, float* bias, float* imgData, int innum, int outnum,  int activation)
{

	float* out = (float*)calloc(outnum, sizeof(float));
	int i, k;
	float sum = 0.0;

	//求nn
	for (k = 0; k < outnum; k++)    //遍历输出列128
	{
		out[k] = 0;
		//进行一次向量乘加运算
		for (i = 0; i < innum; i++)    //遍历权重列1568
		{
			out[k] = out[k] + imgData[i] * weigth[k*innum + i];
			//printf("%f---",out[k]);
		}
		//printf("\n");
	}

	//加偏置
	for (i = 0; i < outnum; i++)    //遍历权重列1568
	{
		out[i] += bias[i];
	}
	
	//relu激活
	if (activation == 1)
	{
		
		for (i = 0; i < outnum; i++)    //遍历权重列1568
		{	//printf("relu激活  ");
			out[i] = (out[i] > 0) ? out[i] : 0;
		}
	}
	else //求softmax激活
	{
		float sum = 0.0;
		for (i = 0; i < outnum; i++)    //求和
		{	//printf("softmax激活  ");
			sum += exp(out[i]);
		}
		for (i = 0; i < outnum; i++)    //求softmax
		{
			out[i] = exp(out[i]) / sum;
		}
	}

	return out;

}

/*im2col_get_pixel(data_im, height, width, channels,im_row, im_col, c_im, pad);
**  从输入的多通道数组im（存储图像数据）中获取指定行、列、、通道数处的元素值
**  输入： im      输入，所有数据存成一个一维数组，例如对于3通道的二维图像而言，
**                每一通道按行存储（每一通道所有行并成一行），三通道依次再并成一行
**        height  每一通道的高度（即输入图像的真正的高度，补0之前）
**        width   每一通道的宽度（即输入图像的宽度，补0之前）
**        channels 输入im的通道数，比如彩色图为3通道，之后每一卷积层的输入的通道数等于上一卷积层卷积核的个数
**        row     要提取的元素所在的行（二维图像补0之后的行数）
**        col     要提取的元素所在的列（二维图像补0之后的列数）
**        channel 要提取的元素所在的通道
**        pad     图像左右上下各补0的长度（四边补0的长度一样）
**  返回： float类型数据，为im中channel通道，row-pad行，col-pad列处的元素值
**  注意：在im中并没有存储补0的元素值，因此height，width都是没有补0时输入图像真正的
**       高、宽；而row与col则是补0之后，元素所在的行列，因此，要准确获取在im中的元素值，
**       首先需要减去pad以获取在im中真实的行列数
*/
float get_pixel(float *im, int inh, int inw,int channel, int row, int col,  int pad)
{
	// 减去补0长度，获取元素在真实图像中的行列数
	row -= pad;
	col -= pad;

	// 如果行列数小于0,或者大于等于输入图像的宽、高，则返回0（刚好是补0的效果）
	if (row < 0 || col < 0 ||
		row >= inh || col >= inh) return 0;
	// im存储多通道二维图像的数据的格式为：各通道所有行并成一行，再多通道依次并成一行，
	// 因此width*height*channel首先移位到所在通道的起点位置，再加上width*row移位到所在指定通道所在行，
	// 最后加上col移位到所在列
	return im[col + inw*(row + inh*channel)];
}

//
//	// 卷积操作可以用旋转180度的特征模板相关来求
//	im2col_cpu(, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
//
//	gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
//
//}
//
///*
//** im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
//**        参数：输入图像、通道数、输入图像高度、宽度、卷积核尺寸、步幅、padding大小、输出的重排数据
//** 输入： data_im	输入图像
//**		 channels	输入图像的通道数（对于第一层，一般是颜色图，3通道，中间层通道数为上一层卷积核个数）
//**		 height 	输入图像的高度（行）
//**		 width		输入图像的宽度（列）
//**		 ksize		卷积核尺寸
//**		 stride 	步幅
//**		 pad		四周补0长度
//**		 data_col	相当于输出，为进行格式重排后的输入图像数据
//** 说明：实际data_col中的元素个数为channels*ksize*ksize*height_col*width_col，
//**      其中channels为data_im的通道数，ksize为卷积核大小，height_col和width_col如下所注。
//**      data_col的行数为channels*ksize*ksize（=每整列包含与某个位置处的卷积核计算的所有通道上的像素）,
//**      （比如输入图像通道数为3,卷积核尺寸为3*3，则共有27行，每列有27个元素），
//**      data_col的列数为height_col*width_col，即一张特征图总的元素个数，不同列对应卷积核在图像上的不同位置做卷积
//*/
//float** im2col_cpu(float* data_im,
//	int channels, int height, int width,
//	int ksize, int stride, int pad, float* data_col)
//{
//	int c, h, w;
//	// 计算该层神经网络的输出图像尺寸（其实没有必要再次计算的，因为在构建卷积层时，make_convolutional_layer()函数
//	// 已经调用convolutional_out_width()，convolutional_out_height()函数求取了这两个参数，
//	// 此处直接使用l.out_h,l.out_w即可，函数参数只要传入该层网络指针就可以）
//	int height_col = (height + 2 * pad - ksize) / stride + 1;     //输出图像的高度
//	int width_col = (width + 2 * pad - ksize) / stride + 1;       //输出图像的宽度
//	/// 卷积核大小：ksize*ksize是一个卷积核的大小，之所以乘以通道数channels，是因为输入图像有多通道，每个卷积核在做卷积时，
//	/// 是同时对同一位置处多通道的图像进行卷积运算。这里为了实现这一目的，将三通道上的卷积核并在一起以便进行计算，因此卷积核
//	/// 实际上并不是二维的，而是三维的。比如对于3通道图像，卷积核尺寸为3*3，该卷积核将同时作用于三通道图像上，这样并起来就得
//	/// 到含有27个元素的卷积核，且这27个元素都是独立的需要训练的参数。所以在计算训练参数个数时，一定要注意每一个卷积核的实际
//	/// 训练参数需要乘以输入通道数。
//	int channels_col = channels * ksize * ksize;　//im2col后的矩阵行数，如3*3*3=27
//
//	//输出重排后的图像行数为channels_col，列数为height_col*width_col       
//	// ******这三层循环之间的逻辑关系，决定了输入图像重排后的格式　*******     
//	// 外循环次数为一个卷积核的尺寸数，循环次数即为最终得到的data_col的总行数    
//	for (c = 0; c < channels_col; ++c) {     //先得到col_data的第一行，，c++得到所有的行。
//		// 卷积核内的列偏移，卷积核是一个二维矩阵，并按行！！！）（一行3列）存储在一维数组中，利用求余运算获取对应在卷积核中的列数，比如对于
//		// 3*3的卷积核（3通道），当c=0时，显然在第一列，当c=5时，显然在第2列，当c=9时，在第二通道上的卷积核的第一列，
//		// 当c=26时，在第三列（第三通道上）
//		int w_offset = c % ksize;    //除以3求余数就是卷积核大小内的列偏移
//
//		// 卷积核大小内的行偏移，卷积核是一个二维的矩阵，且是按行（卷积核所有行并成一行）存储在一维数组中的，
//		// 比如对于3*3的卷积核，处理3通道的图像，那么一个卷积核具有27个元素，每9个元素对应一个通道上的卷积核（互为一样），
//		// 每当c为3的倍数，就意味着卷积核换了一行，h_offset取值为0,1,2，对应3*3卷积核中的第1, 2, 3行
//		int h_offset = (c / ksize) % ksize;    //得到对应一通道上的行数
//
//		// 通道偏移（在卷积核中的），channels_col是多通道的卷积核并在一起的，比如对于3通道，3*3卷积核，每过9个元素就要换一通道数，
//		// 当c=0~8时，c_im=0;　c=9~17时，c_im=1;　c=18~26时，c_im=2
//		int c_im = c / ksize / ksize;　 //计算目前处理图像的第几个通道
//
//		// 中循环次数等于该层输出图像行数height_col，说明data_col中的每一行存储了一张特征图，这张特征图又是按行存储在data_col中的某行中
//		for (h = 0; h < height_col; ++h) {
//			// 内循环等于该层输出图像列数width_col，说明最终得到的data_col总有channels_col行，height_col*width_col列
//			for (w = 0; w < width_col; ++w) {
//				// 由上面可知，对于3*3的卷积核，h_offset取值为0,1,2,当h_offset=0时，会提取出所有与卷积核第一行元素进行运算的像素，
//				// 依次类推；加上h*stride是对卷积核进行行移位操作，比如卷积核从图像(0,0)位置开始做卷积，那么最先开始涉及(0,0)~(3,3)
//				// 之间的像素值，若stride=2，那么卷积核进行一次行移位时，下一行的卷积操作是从元素(2,0)（2为图像行号，0为列号）开始
//				int im_row = h_offset + h * stride;
//
//				// 对于3*3的卷积核，w_offset取值也为0,1,2，当w_offset取1时，会提取出所有与卷积核中第2列元素进行运算的像素，
//				// 实际在做卷积操作时，卷积核对图像逐行扫描做卷积，加上w*stride就是为了做列移位，
//				// 比如前一次卷积其实像素元素为(0,0)，若stride=2,那么下次卷积元素起始像素位置为(0,2)（0为行号，2为列号）
//				int im_col = w_offset + w * stride;
//
//				// col_index为重排后图像中的像素索引，等于c * height_col * width_col + h * width_col +w（还是按行存储，所有通道再并成一行），
//				// 对应第c通道，h行，w列的元素
//				int col_index = (c * height_col + h) * width_col + w;
//
//				// im2col_get_pixel函数获取输入图像data_im中第c_im通道，im_row,im_col的像素值并赋值给重排后的图像，
//				// height和width为输入图像data_im的真实高、宽，pad为四周补0的长度（注意im_row,im_col是补0之后的行列号，
//				// 不是真实输入图像中的行列号，因此需要减去pad获取真实的行列号）
//				data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
//					im_row, im_col, c_im, pad);
//			}
//		}
//	}
//
//}


//
//
///*	gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
//**  参数：  A,C的行数M、B,C的列数N、ALPGA=1,权重A、A的列、col_data B、B的列数、卷积输出C、C的列数
//**	功能：被gemm_cpu()函数调用，实际完成C = ALPHA * A * B + C 矩阵计算，
//**		 输出的C也是按行存储（所有行并成一行）
//**	输入： A,B,C   输入矩阵（一维数组格式）
//**		  ALPHA   系数
//**		  BETA	  系数
//**		  M 	  A,C的行数（不做转置）或者A'的行数（做转置），此处A未转置，故为A的行数
//**		  N 	  B,C的列数（不做转置）或者B'的列数（做转置），此处B未转置，故为B的列数
//**		  K 	  A的列数（不做转置）或者A'的列数（做转置），B的行数（不做转置）或者B'的行数（做转置），此处A,B均未转置，故为A的列数、B的行数
//**		  lda	  A的列数（不做转置）或者A'的行数（做转置），此处A未转置，故为A的列数
//**		  ldb	  B的列数（不做转置）或者B'的行数（做转置），此处B未转置，故为B的列数
//**		  ldc	  C的列数
//**	说明1：此函数是用C实现矩阵乘法运算，这部分代码模仿Caffe中的math_functions.cpp的代码
//**		 参考博客：http://www.voidcn.com/blog/thy_2014/article/p-6149690.html
//**		 更为详细的注释参见：gemm_cpu()函数的注释
//**	说明2：此函数在gemm_cpu()函数中调用，是其中四种情况之一，A,B都不进行转置
//**		 函数名称gemm_nn()中的两个nn分别表示not transpose， not transpose
//*/
//
//float** gemm_nn(int M, int N, int K, float ALPHA,
//	float *A, int lda,
//	float *B, int ldb,
//	float *C, int ldc)
//{	// input: 矩阵A[M,K], filter: 矩阵B[K,N],  output: 矩阵C[M,N]
//	int i, j, k;
//#pragma omp parallel for
//	// 大循环：遍历输出C(输入权重A)的每一行，i表示A的第i行，也是C的第i行
//	for (i = 0; i < M; ++i){
//		// 中循环：遍历输入权重A每一行的所有列，k表示权重A的第k列，同时表示B的第k行
//		for (k = 0; k < K; ++k){
//			// 先计算ALPHA * A（A中每个元素乘以ALPHA）
//			register float A_PART = ALPHA*A[i*lda + k];
//			// 内循环：遍历B（输出C）中所有列，每次大循环完毕，将计算得到A*B一行的结果
//			// j是B的第j列，也是C的第j列
//			for (j = 0; j < N; ++j){
//				// A中的第i行k列与B中的k行j列对应相乘，因为一个大循环要计算A*B整行之结果，
//				// 因此，这里用了一个内循环，并没有直接乘以B[k*ldb+i]
//				// 每个内循环完毕，将计算A*B整行的部分结果（A中第i行k列!与B所有列第k行!所有元素相乘的结果
//				C[i*ldc + j] += A_PART*B[k*ldb + j];
//			}
//		}
//	}
//}





///*
//#define BUFFER_SIZE 2
//合并文件mergeFile (infile1, infile2)
//void mergeFile(const char *fp1, const char *fp2)
//{
//	FILE *fd1, *fd2;           //创建两个文件指针
//	float buf[1];
//
//	int rc1, i=0;
//	fd1 = fopen(fp1, "rb");   //打开源文件
//	fd2 = fopen(fp2, "ab");   //打开目的文件
//
//	while ((rc1 = fread(buf, sizeof(float), 1, fd1)) != 0)    //读取源文件中的一个float数据，复制到buf
//	{
//		i++;
//		fwrite(buf, sizeof(float), rc1, fd2);                 //将复制的buf中的数据，拷贝到目的文件中
//	}
//	printf("%d\n",i);
//	
//	Sleep(0.1);              //等待拷贝
//	fclose(fd1);             //关闭源文件
//	fclose(fd2);			 //关闭目的文件
//
//}
//*/
///*
//	对卷积神经网络cnn进行初始化,
//	卷积核的大小为3x3,stride=1,same padding 
//	池化核大小2x2，stride=2
//	卷积神经网络的初始化主要包含了创建各数据的内存空间及权重的随机赋值，按照结构分配空间就可以了
//	cnnsetup(cnn,inputSize,outSize)，参数：CNN整个网络结构体、网络输入的宽，高、网络输出的维度
//
//void cnnsetup(CNN* cnn,nSize inputSize,int outputSize)
//{
//	cnn->layerNum=6;    //网络的总层数为5
//
//	nSize inSize;         //包含高、宽值的结构体
//	int mapSize=3;              //卷积核的大小3x3
//
//	c1卷积层的初始化，返回初始化好的卷积层结构体
//	参数：输入宽、输入高、卷积核尺寸、输入通道数、输出通道数(stride=1,same padding)
//	inSize.c=inputSize.c;       //C1层输入的宽
//	inSize.r=inputSize.r;       //C1层输入的行
//	cnn->C1=initCovLayer(inSize.c,inSize.r,3,1,16);								 //初始化C1层
//
//	s2池化层的初始化，返回初始化好的池化层
//	参数：输入宽、输入高、池化核大小、输入通道数、输出通道数、池化类型
//	inSize.c=inSize.c;			 //C1层输出的宽=S2层输入的宽
//	inSize.r=inSize.r;			 //C1层输出的高=S2层输入的宽
//	cnn->S2=initPoolLayer(inSize.c,inSize.r,2,16,16,MaxPool);					//初始化S2层
//
//	c3卷积层的初始化，返回初始化好的卷积层结构体
//	        参数：输入宽、输入高、卷积核尺寸、输入通道数、输出通道数
//	inSize.c=inSize.c/2;		 //S2层的输出高=C3层输入的宽=n/2
//	inSize.r=inSize.r/2;		 //S2层的输出宽=C3层输入的行=n/2
//	cnn->C3=initCovLayer(inSize.c,inSize.r,3,16,32);							//初始化C3层
//
//	s4池化层的初始化，返回初始化好的池化层
//	inSize.c=inSize.c;			 //S4层输入的宽
//	inSize.r=inSize.r;			 //S4层输入的行
//	cnn->S4=initPoolLayer(inSize.c,inSize.r,2,32,32,MaxPool);					 //初始化S4层
//
//	inSize.c=inSize.c/2;         //05层输入的宽=n/2
//	inSize.r=inSize.r/2;         //05层输入的行=n/2
//	05输出全连接层的初始化
//	参数：	     输入神经元个数、输出神经元个数
//	cnn->O5=initOutLayer(inSize.c*inSize.r*32,128);								 //初始化05层
//
//	05输出全连接层的初始化
//	cnn->O6=initOutLayer(128, outputSize);									 //初始化06层
//	cnn->e=(float*)calloc(cnn->O5->outputNum,sizeof(float));					 //训练误差
//}
//*/
///*
//cnn层的初始化
//cnn->C1=initCovLayer(inSize.c,inSize.r,3,1,16);    入道1，出道6				  //初始化CU层
//cnn->C3=initCovLayer(inSize.c,inSize.r,3,16,32);	入道6，出道12				 //初始化C3层
//7参数：         输入宽、输入高、卷积核尺寸、输入通道数、输出通道数，padding大小，步幅大小
//CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels,int pad,int stride)
//{
//	CovLayer* covL=(CovLayer*)malloc(sizeof(CovLayer));    //新建卷积层的结构体
//
//	
//	covL->inputHeight=inputHeight;                    //定义卷积层的输入图像高
//	covL->inputWidth=inputWidth;					  //定义卷积层的输入图像宽
//	covL->mapSize=mapSize;							  //定义卷积层的卷积核大小
//	covL->inChannels=inChannels;					  //定义卷积层的输入通道数
//	covL->outChannels=outChannels;                    //定义卷积层的输出通道数
//	covL->pad = pad;								  //定义卷积层的输入padding大小
//	covL->stride = stride;							  //定义卷积层的步幅大小
//
//	covL->isFullConnect=true;						 // 默认为全连接
//
//	 创建权重的内存空间（--4维float型），按行存储
//	int i,j,c,r;
//	srand((unsigned)time(NULL)); //设置随机数种子的函数，以现在的系统时间作为随机数的种子来产生随机数
//								 rand（）是一个产生随机数的函数，两个函数配合使用
//
//	covL->mapData=(float****)malloc(outChannels*sizeof(float***));  //分配卷积深度空间
//	for(i=0;i<outChannels;i++){   //分配n个卷积空间
//		covL->mapData[i]=(float***)malloc(outChannels*sizeof(float**));
//		for(j=0;j<inChannels;j++){  //一下两个循环分配一张特征图的大小
//			covL->mapData[i][j]=(float**)malloc(mapSize*sizeof(float*));
//			for(r=0;r<mapSize;r++){
//				covL->mapData[i][j][r]=(float*)malloc(mapSize*sizeof(float));
//				for(c=0;c<mapSize;c++){
//					float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2; 
//					covL->mapData[i][j][r][c]=randnum*sqrt((float)6.0/(float)(mapSize*mapSize*(inChannels+outChannels)));
//				}
//			}
//		}
//	}
//
//
//	创建偏置的内存空间（--1维float型）
//	covL->basicData=(float*)calloc(outChannels,sizeof(float));
//
//	创建输出特征图的内存空间（经过激活函数的和未经过激活函数的）--3维
//	卷积层输出的尺寸大小
//	int outW=inputWidth;
//	int outH=inputHeight;
//	
//	covL->v=(float***)malloc(outChannels*sizeof(float**));  //为激活前的值分配空间（3维float型）
//	covL->y=(float***)malloc(outChannels*sizeof(float**));  //为激活后的值分配空间（3维float型）
//	for(j=0;j<outChannels;j++){   
//		covL->d[j]=(float**)malloc(outH*sizeof(float*));     
//		covL->v[j]=(float**)malloc(outH*sizeof(float*));      
//		covL->y[j]=(float**)malloc(outH*sizeof(float*));      
//		for(r=0;r<outH;r++){
//			covL->d[j][r]=(float*)calloc(outW,sizeof(float));
//			covL->v[j][r]=(float*)calloc(outW,sizeof(float));      
//			covL->y[j][r]=(float*)calloc(outW,sizeof(float));
//		}
//	}
//
//	return covL;      //返回初始化好的卷积层
//}
//*/
//
///*
//cnn->S2=initPoolLayer(inSize.c,inSize.r,2,16,16,MaxPool)     //初始化S2层
//cnn->S4=initPoolLayer(inSize.c,inSize.r,2,32,32,MaxPool);  //初始化S4层
//池化层的初始化：
//6参数：   输入宽、输入高、池化核大小、入道、出道、池化类型
//PoolLayer* initPoolLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels,int poolType)
//{
//	PoolLayer* poolL=(PoolLayer*)malloc(sizeof(PoolLayer));    //新建池化层的结构体
//
//	定义池化层的输入高、输入宽、池化核大小、输入通道数、输出通道数、池化类型
//	poolL->inputHeight=inputHeight;                  //定义池化层的输入高
//	poolL->inputWidth=inputWidth;                    //定义池化层的输入高
//	poolL->mapSize=mapSize;                          //定义池化核大小
//	poolL->inChannels=inChannels;                    //定义池化层的输入通道数
//	poolL->outChannels=outChannels;                  //定义池化层的输出通道数
//	poolL->poolType=poolType;                        //池化类型
//
//	创建偏置的内存空间（--1维float型）
//	poolL->basicData=(float*)calloc(outChannels,sizeof(float));
//	
//	创建输出特征图的三维内存空间
//	int outW=inputWidth/mapSize;  
//	int outH=inputHeight/mapSize;
//	int j,r;
//	poolL->d=(float***)malloc(outChannels*sizeof(float**));    // 网络的局部梯度,δ值
//	poolL->y=(float***)malloc(outChannels*sizeof(float**));    // 采样函数后神经元的输出（3维float型）,无激活函数，分配空间
//	for(j=0;j<outChannels;j++){
//		poolL->d[j]=(float**)malloc(outH*sizeof(float*));      
//		poolL->y[j]=(float**)malloc(outH*sizeof(float*)); 
//		for(r=0;r<outH;r++){
//			poolL->d[j][r]=(float*)calloc(outW,sizeof(float));
//			poolL->y[j][r]=(float*)calloc(outW,sizeof(float));
//		}
//	}
//
//	return poolL;   //返回初始化好的池化层
//}
//*/
///*
//输出全连接层的初始化
// cnn->O5 = initOutLayer(inSize.c*inSize.r * 12, outputSize);    //初始化05层
//参数：				输入神经元个数、输出神经元个数
//OutLayer* initOutLayer(int inputNum,int outputNum)
//{
//	OutLayer* outL=(OutLayer*)malloc(sizeof(OutLayer));      //新建输出层的结构体
//
//	定义输出全连接层的输入数据数目、输出数据数目、偏置、
//	outL->inputNum=inputNum;                     //定义输出全连接层的输入数据数目          
//	outL->outputNum=outputNum;                   //定义输出全连接层的输出数据数目
//
//	创建偏置的分配空间（float型）
//	outL->basicData=(float*)calloc(outputNum,sizeof(float));
//
//	 创建全连接层输出的空间空间（1维float型）
//	outL->d=(float*)calloc(outputNum,sizeof(float));
//	outL->v=(float*)calloc(outputNum,sizeof(float));   // 为进入激活函数的输入值分配空间
//	outL->y=(float*)calloc(outputNum,sizeof(float));   // 为激活函数后神经元的输出分配空间
//
//	 创建全连接层权重的初始化分配空间（2维float型）
//	outL->wData=(float**)malloc(outputNum*sizeof(float*)); // 输入行，输出列
//	int i,j;
//	srand((unsigned)time(NULL));
//	for(i=0;i<inputNum;i++){
//		outL->wData[i]=(float*)malloc(inputNum*sizeof(float));  // 为全连接层权重的初始化分配空间（float型）
//		for(j=0;j<outputNum;j++){
//			float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2; // 产生一个-1到1的随机数
//			outL->wData[i][j]=randnum*sqrt((float)6.0/(float)(inputNum+outputNum));
//		}
//	}
//
//	outL->isFullConnect=true; 
//
//	return outL;  //返回初始化好的输出层
//}
//
//*/
//(vecmaxIndex(cnn->O5->y,cnn->O5->outputNum)
//int vecmaxIndex(float* vec, int veclength)// 返回向量最大数的序号
//{
//	int i;
//	float maxnum=-1.0;
//	int maxIndex=0;
//	for(i=0;i<veclength;i++){
//		if(maxnum<vec[i]){
//			maxnum=vec[i];
//			maxIndex=i;
//		}
//	}
//	return maxIndex;
//}
//
// 测试cnn函数
// incorrectRatio=cnntest(cnn,testImg,testLabel,testNum);
//参数：         CNN网络结构体、测试数据的结构体、测试标签的结构体、测试数目（1）
//float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum)
//{
//	int n=0,outindex;
//	int incorrectnum=0;  //错误预测的数目
//	for(n=0;n<testNum;n++){        //遍历测试数目
//		cnnff(cnn,inputData->ImgPtr[n].ImgData);    //对第n张图片数据进行前向传播
//		/*
//		if(vecmaxIndex(cnn->O5->y,cnn->O5->outputNum)!=vecmaxIndex(outputData->LabelPtr[n].LabelData,cnn->O5->outputNum))
//			incorrectnum++;
//		*/
//		outindex = vecmaxIndex(cnn->O6->y, cnn->O6->outputNum);  //输出最后一层的输出结果
//		printf("输出的结果：%d", outindex);
//		cnnclear(cnn);
//		printf("testing: %d\n", n);    //打印测试的轮次
//	}
//	return (float)incorrectnum/(float)testNum;
//}
//
//
///*
// 导入训练好的cnn的权重数据filename文件到cnn网络中
//void importcnn(CNN* cnn,   char* filename)
//{	
//	打开权重文件
//	char* filedir = "D:\\FPGA-AI\\Lenet_MNIST\\DeepLearningC-master\\CNN\\CNNdata";
//	const char* allweigth_filename = filename;   //将权重文件路径转为，一个指向字符串常量的指针
//	const char* c1_filename = "D:\\FPGA-AI\\Lenet_MNIST\\DeepLearningC-master\\CNN\\weight_bin\\c1w.bin";
//	FILE  *fp=NULL;									//创建一个文件指针
//	fp = fopen(allweigth_filename, "rb");          //打开训练好的网络权重文件allweigth_file.bin
//	if (fp == NULL)
//		printf("write file failed2\n");
//	else printf("read all_weight ok!\n");
//
//	 读取C1的卷积权重（4维float型数组）
//	int i,j,c,r;
//	
//	for(i=0;i<cnn->C1->outChannels;i++)
//		for(j=0;j<cnn->C1->inChannels;j++)
//			for(r=0;r<cnn->C1->mapSize;r++)
//				for(c=0;c<cnn->C1->mapSize;c++){
//					float* in=(float*)malloc(sizeof(float));   //分配一个float型权重的内存空间
//					fread(in, sizeof(float), 1, fp);           //从权重文件中读取一个float字节的权重到in指针
//					cnn->C1->mapData[i][j][r][c]=*in;		   //将读取的一个权值in赋值给网络的权重mapData
//				}
//	printf("import c1 weight ok!--");
//	fclose(fp);
//
//	
//	读取 C1的偏置（1维float型数组）
//	for(i=0;i<cnn->C1->outChannels;i++)
//		fread(&cnn->C1->basicData[i], sizeof(float), 1, fp);
//	printf("import c1b ok!--");
//	
//
//	
//	 读取C3网络的卷积权重（4维float型数组）
//	for(i=0;i<cnn->C3->outChannels;i++)
//		for(j=0;j<cnn->C3->inChannels;j++)
//			for(r=0;r<cnn->C3->mapSize;r++)
//				for(c=0;c<cnn->C3->mapSize;c++)
//				fread(&cnn->C3->mapData[i][j][r][c],sizeof(float),1,fp);
//	printf("import c3 weight ok!--");
//	读取C3的偏置（1维float型数组）
//	for(i=0;i<cnn->C3->outChannels;i++)
//		fread(&cnn->C3->basicData[i],sizeof(float),1,fp);
//	printf("import c3b ok!--");
//
//	 读取f5输出层的卷积权重（2维float型数组）
//	for(i=0;i<cnn->O5->outputNum;i++)
//		for(j=0;j<cnn->O5->inputNum;j++)
//			fread(&cnn->O5->wData[i][j],sizeof(float),1,fp);
//	printf("import f5w ok!--");
//	读取O5输出层的偏置（1维float型数组）
//	for(i=0;i<cnn->O5->outputNum;i++)
//		fread(&cnn->O5->basicData[i],sizeof(float),1,fp);
//	printf("import f5b ok!--");
//
//	 读取f6输出层的卷积权重（2维float型数组）
//	for (i = 0; i<cnn->O6->outputNum; i++)
//		for (j = 0; j<cnn->O6->inputNum; j++)
//			fread(&cnn->O6->wData[i][j], sizeof(float), 1, fp);
//	printf("import f6w ok!--");
//	读取O6输出层的偏置（1维float型数组）
//	for (i = 0; i<cnn->O6->outputNum; i++)
//		fread(&cnn->O6->basicData[i], sizeof(float), 1, fp);
//	printf("import f6b ok!--");
//	
//	
//	
//}
//*/
//
//
//
//
//
// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
//float activation_Sigma(float input,float bas) // sigma激活函数
//{
//	float temp=input+bas;
//	return (float)1.0/((float)(1.0+exp(-temp)));
//}
//
//
//平均池化
//void avgPooling(float** output,nSize outputSize,float** input,nSize inputSize,int mapSize) // 求平均值
//{
//	int outputW=inputSize.c/mapSize;
//	int outputH=inputSize.r/mapSize;
//	if(outputSize.c!=outputW||outputSize.r!=outputH)
//		printf("ERROR: output size is wrong!!");
//
//	int i,j,m,n;
//	for(i=0;i<outputH;i++)
//		for(j=0;j<outputW;j++)
//		{
//			float sum=0.0;
//			for(m=i*mapSize;m<i*mapSize+mapSize;m++)
//				for(n=j*mapSize;n<j*mapSize+mapSize;n++)
//					sum=sum+input[m][n];
//
//			output[i][j]=sum/(float)(mapSize*mapSize);
//		}
//}
//
// 单层全连接神经网络的前向传播
//float vecMulti(float* vec1,float* vec2,int vecL)// 两向量相乘
//{
//	int i;
//	float m=0;
//	for(i=0;i<vecL;i++)
//		m=m+vec1[i]*vec2[i];
//	return m;
//}
//
//全连接的前向传播
//void nnff(float* output,float* input,float** wdata,float* bas,nSize nnSize)
//{
//	int w=nnSize.c;
//	int h=nnSize.r;
//	
//	int i;
//	for(i=0;i<h;i++)
//		output[i]=vecMulti(input,wdata[i],w)+bas[i];
//}
//
//float sigma_derivation(float y){ // Logic激活函数的自变量微分
//	return y*(1-y); // 这里y是指经过激活函数的输出值，而不是自变量
//}
//
//
//
//
//
//网络的中间数据清零，只留下权重数据
//void cnnclear(CNN* cnn)
//{
//	 将神经元的部分数据清除
//	int j,c,r;
//	 C1网络
//	for(j=0;j<cnn->C1->outChannels;j++){
//		for(r=0;r<cnn->S2->inputHeight;r++){
//			for(c=0;c<cnn->S2->inputWidth;c++){
//				cnn->C1->d[j][r][c]=(float)0.0;
//				cnn->C1->v[j][r][c]=(float)0.0;
//				cnn->C1->y[j][r][c]=(float)0.0;
//			}
//		}
//	}
//	 S2网络
//	for(j=0;j<cnn->S2->outChannels;j++){
//		for(r=0;r<cnn->C3->inputHeight;r++){
//			for(c=0;c<cnn->C3->inputWidth;c++){
//				cnn->S2->d[j][r][c]=(float)0.0;
//				cnn->S2->y[j][r][c]=(float)0.0;
//			}
//		}
//	}
//	 C3网络
//	for(j=0;j<cnn->C3->outChannels;j++){
//		for(r=0;r<cnn->S4->inputHeight;r++){
//			for(c=0;c<cnn->S4->inputWidth;c++){
//				cnn->C3->d[j][r][c]=(float)0.0;
//				cnn->C3->v[j][r][c]=(float)0.0;
//				cnn->C3->y[j][r][c]=(float)0.0;
//			}
//		}
//	}
//	 S4网络
//	for(j=0;j<cnn->S4->outChannels;j++){
//		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
//			for(c=0;c<cnn->S4->inputWidth/cnn->S4->mapSize;c++){
//				cnn->S4->d[j][r][c]=(float)0.0;
//				cnn->S4->y[j][r][c]=(float)0.0;
//			}
//		}
//	}
//	 O5输出
//	for(j=0;j<cnn->O5->outputNum;j++){
//		cnn->O5->d[j]=(float)0.0;
//		cnn->O5->v[j]=(float)0.0;
//		cnn->O5->y[j]=(float)0.0;
//	}
//}
//
// 这是用于测试的函数
//void savecnndata(CNN* cnn,const char* filename,float** inputdata) // 保存CNN网络中的相关数据
//{
//	FILE  *fp=NULL;
//	fp=fopen(filename,"wb");
//	if(fp==NULL)
//		printf("write file failed3\n");
//
//	 C1的数据
//	int i,j,r;
//	 C1网络
//	for(i=0;i<cnn->C1->inputHeight;i++)
//		fwrite(inputdata[i],sizeof(float),cnn->C1->inputWidth,fp);
//	for(i=0;i<cnn->C1->inChannels;i++)
//		for(j=0;j<cnn->C1->outChannels;j++)
//			for(r=0;r<cnn->C1->mapSize;r++)
//				fwrite(cnn->C1->mapData[i][j][r],sizeof(float),cnn->C1->mapSize,fp);
//
//	fwrite(cnn->C1->basicData,sizeof(float),cnn->C1->outChannels,fp);
//
//	for(j=0;j<cnn->C1->outChannels;j++){
//		for(r=0;r<cnn->S2->inputHeight;r++){
//			fwrite(cnn->C1->v[j][r],sizeof(float),cnn->S2->inputWidth,fp);
//		}
//		for(r=0;r<cnn->S2->inputHeight;r++){
//			fwrite(cnn->C1->d[j][r],sizeof(float),cnn->S2->inputWidth,fp);
//		}
//		for(r=0;r<cnn->S2->inputHeight;r++){
//			fwrite(cnn->C1->y[j][r],sizeof(float),cnn->S2->inputWidth,fp);
//		}
//	}
//
//	 S2网络
//	for(j=0;j<cnn->S2->outChannels;j++){
//		for(r=0;r<cnn->C3->inputHeight;r++){
//			fwrite(cnn->S2->d[j][r],sizeof(float),cnn->C3->inputWidth,fp);
//		}
//		for(r=0;r<cnn->C3->inputHeight;r++){
//			fwrite(cnn->S2->y[j][r],sizeof(float),cnn->C3->inputWidth,fp);
//		}
//	}
//	 C3网络
//	for(i=0;i<cnn->C3->inChannels;i++)
//		for(j=0;j<cnn->C3->outChannels;j++)
//			for(r=0;r<cnn->C3->mapSize;r++)
//				fwrite(cnn->C3->mapData[i][j][r],sizeof(float),cnn->C3->mapSize,fp);
//
//	fwrite(cnn->C3->basicData,sizeof(float),cnn->C3->outChannels,fp);
//
//	for(j=0;j<cnn->C3->outChannels;j++){
//		for(r=0;r<cnn->S4->inputHeight;r++){
//			fwrite(cnn->C3->v[j][r],sizeof(float),cnn->S4->inputWidth,fp);
//		}
//		for(r=0;r<cnn->S4->inputHeight;r++){
//			fwrite(cnn->C3->d[j][r],sizeof(float),cnn->S4->inputWidth,fp);
//		}
//		for(r=0;r<cnn->S4->inputHeight;r++){
//			fwrite(cnn->C3->y[j][r],sizeof(float),cnn->S4->inputWidth,fp);
//		}
//	}
//
//	 S4网络
//	for(j=0;j<cnn->S4->outChannels;j++){
//		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
//			fwrite(cnn->S4->d[j][r],sizeof(float),cnn->S4->inputWidth/cnn->S4->mapSize,fp);
//		}
//		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
//			fwrite(cnn->S4->y[j][r],sizeof(float),cnn->S4->inputWidth/cnn->S4->mapSize,fp);
//		}
//	}
//
//	 O5输出层
//	for(i=0;i<cnn->O5->outputNum;i++)
//		fwrite(cnn->O5->wData[i],sizeof(float),cnn->O5->inputNum,fp);
//	fwrite(cnn->O5->basicData,sizeof(float),cnn->O5->outputNum,fp);
//	fwrite(cnn->O5->v,sizeof(float),cnn->O5->outputNum,fp);
//	fwrite(cnn->O5->d,sizeof(float),cnn->O5->outputNum,fp);
//	fwrite(cnn->O5->y,sizeof(float),cnn->O5->outputNum,fp);
//
//	fclose(fp);
//}
