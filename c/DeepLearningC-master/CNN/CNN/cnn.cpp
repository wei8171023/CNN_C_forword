#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include <windows.h>
#include "cnn.h"


//������㣬
//�������ò��Ȩ��ָ�룬ƫ��ָ�룬����ͼ������ָ�롢���������ߡ�����ͨ���������ͨ����������˵Ĵ�С��������С�����߷�1�ǽ��в���,0�ǲ����ߣ�,�������1��relu,0��softmax��
float* conv(float* weigth, float* bias, float* imgData, int inw, int inh, int inch, int outchan, int fsize, int stride, int pad, int activation)
{
	
	int outw = (inw + 2 * pad - fsize) / stride + 1;						//�����
	int outh = (inh + 2 * pad - fsize) / stride + 1;						//�����
	printf("---��������%d���ߣ�%d\n",outw,outh);

	
	//����1άfloat���������ͼָ�룬��δָ����䶯̬�ڴ�ռ䣬�ÿռ�洢float�͵�����;
	float* out = (float*)calloc(outw*outh*outchan, sizeof(float));

	int pad_w, pad_h;                      //����padding���ͼ�����
	//int filter_size=fsize*fsize*num_filter;					       //�����Ԫ�ظ���
	
	//����Paddingģʽ��1ΪSAME,0Ϊvaild��������padding���ͼ�����
	pad_w = inw + pad*2*(fsize-1)/2;              //padding��Ŀ�
	pad_h = inh + pad*2*(fsize - 1) / 2;          //padding��ĸ�
	printf("padding�����%d���ߣ�%d\n", pad_w, pad_h);
	//system("pause");

	//����1άfloat��padding������ͼָ�룬��Ϊָ����䶯̬�ڴ�ռ䣬�ÿռ�洢float�͵�����
	float* img_pad = (float*)calloc(pad_h*pad_w*inch, sizeof(float));   
	if (pad == 1)
	{
		printf("---��ʼpadding \n");
		int p, q,r,a=0;
		for (r = 0; r < inch; r++)    //��������ͨ��
		{
			for (p = 0; p < pad_h; p++)    //����pad��
			{
				for (q = 0; q < pad_w; q++)    //����pad��
				{
					a++;
					img_pad[r*pad_h*pad_w + p*pad_w + q]=get_pixel(imgData,inh,inw,r,p,q,pad);
				}
			}
				
		}
		//printf("padding������������%d   ",a);   //900��
		printf("---padding ��ɣ�\n");
	}
	else 
	{
		img_pad = imgData;
	}
  
	printf("---��ʼ������� \n");
	int i, j,k,l,m,n,col=0,row=0;
	float sum = 0.0;
	#pragma omp parallel for
	for (k = 0; k < outchan; k++)    //�������ͨ��==����˸���
	{
		//printf("k==%d====================================================================\n ", k);
		#pragma omp parallel for 
		for (i = 0; i < outh; i++)         //���������
		{
			//printf("i==%d=====================================================\n ", i);
			#pragma omp parallel for 
			for (j = 0; j < outw; j++)          //���������
			{ 
				out[k*outh*outw + i*outw + j] = 0;
				//����һ�ξ��������һ��ƫ����Ӳ���
				//printf("j==%d===================================\n ", j);
				for (l = 0; l < inch; l++)            //��������ͨ��
				{
					//printf("l==%d======================\n ", l);
					#pragma omp parallel for
					for (m = 0; m < fsize; m++)         //��������˵ĸ�
					{
						//printf("m==%d=============\n ", m);
						#pragma omp parallel for
						for (n = 0; n < fsize; n++)       //��������˵Ŀ�
						{
							//printf("n==%d********\n ", n);
							row = stride*i + m;   //���γ˷���pad�ϵ���=���ξ����pad�ϵ���ʼ��+������ƫ��
							//printf("�˼��У�%d ", row);
							col = stride*j + n;   //���γ˷���pad�ϵ���=���ξ����pad�ϵ���ʼ��+������ƫ��
							//printf("�˼��У�%d ", col);
							out[k*outh*outw + i*outw + j] += weigth[k*fsize*fsize*inch + l*fsize*fsize + m*fsize + n] * img_pad[l*pad_h*pad_w + row*pad_w + col];
						}
					}
				}
				//printf("�˼ӽ����%f\n", out[k*outh*outw + i*outw + j]);
				//��ƫ��
				out[k*outh*outw + i*outw + j] += bias[k];   
				if (activation == 1)   //relu����(���������)
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
					sum += exp(out[k* outh* outw + i* outw + j]);   //��softmax��ĸ֮��
				}
			}
		}
	}

	if (activation == 0)  //softmax������ȫ���Ӳ��У�
	{
		int x, y, z;

		for (z = 0; z < outchan; z++)    //�������ͨ��==����˸���
		{
			for (x = 0; x < outh; x++)         //���������
			{
				for (y = 0; y < outw; y++)          //���������
				{
					out[z* outh * outw + x* outw + y] = exp(out[z* outh * outw + x* outw + y])/sum; //�����ÿ�������softmaxֵ
				}
			}
		}
	}
	else
	{
		NULL;
	}
	
	printf("---����������\n");
	return out;
}

//���ػ�����
//������         ����ͼ������ߡ����������ͨ�������ػ��˴�С��������С
float* maxpool(float* imgData, int inh, int inw, int inchan, int psize, int stride)
{
	int outh, outw, outchan;
	outh = (inh - psize) / stride + 1;   //�ػ������
	outw = (inw - psize) / stride + 1;   //�ػ������
	outchan = inchan;                    //���ͨ����
	printf("---�ػ������%d���ߣ�%d\n", outw, outh);
	//����1άfloat���������ͼָ�룬��Ϊָ����䶯̬�ڴ�ռ䣬�ÿռ�洢float�͵�����;
	float* out = (float*)calloc(outw*outh*outchan, sizeof(float));

	int i, j, k,m,n,col,row;
	float max;
	#pragma omp parallel for
	for (k = 0; k < outchan; k++)     //�������ͨ��==����˸���
	{
		//printf("k==%d====================================================================\n ", k);
		#pragma omp parallel for
		for (i = 0; i < outh; i++)         //���������
		{
			//printf("i==%d=====================================================\n ", i);
			#pragma omp parallel for
			for (j = 0; j < outw; j++)          //���������
			{
				//printf("j==%d===================================\n ", j);
				//����һ�γػ�����
				max = 0.0;
				//������ͨ��һ���ػ��ˣ���ȡ���е����ֵ
				for (m = 0; m < psize; m++)   //�����ػ��˵ĸ�
				{
					//printf("m==%d=============\n ", m);
					for (n = 0; n < psize; n++)    //�����ػ��˵Ŀ�
					{
						//printf("n==%d********\n ", n);
						row = stride*i + m;   //���γػ�������ͼ���ϵ���=���γػ�������ͼ���ϵ���ʼ��+����ƫ��
						col = stride*j + n;   //���γػ�������ͼ���ϵ���=���γػ�������ͼ���ϵ���ʼ��+����ƫ��
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
	printf("---���ػ��������\n");
	return out;
}

//ȫ��������
//������ �ò��Ȩ��ָ�룬ƫ��ָ�룬����ͼ������ָ�룬Ȩ��������       Ȩ��������     ��������
float* nn(float* weigth, float* bias, float* imgData, int innum, int outnum,  int activation)
{

	float* out = (float*)calloc(outnum, sizeof(float));
	int i, k;
	float sum = 0.0;

	//��nn
	for (k = 0; k < outnum; k++)    //���������128
	{
		out[k] = 0;
		//����һ�������˼�����
		for (i = 0; i < innum; i++)    //����Ȩ����1568
		{
			out[k] = out[k] + imgData[i] * weigth[k*innum + i];
			//printf("%f---",out[k]);
		}
		//printf("\n");
	}

	//��ƫ��
	for (i = 0; i < outnum; i++)    //����Ȩ����1568
	{
		out[i] += bias[i];
	}
	
	//relu����
	if (activation == 1)
	{
		
		for (i = 0; i < outnum; i++)    //����Ȩ����1568
		{	//printf("relu����  ");
			out[i] = (out[i] > 0) ? out[i] : 0;
		}
	}
	else //��softmax����
	{
		float sum = 0.0;
		for (i = 0; i < outnum; i++)    //���
		{	//printf("softmax����  ");
			sum += exp(out[i]);
		}
		for (i = 0; i < outnum; i++)    //��softmax
		{
			out[i] = exp(out[i]) / sum;
		}
	}

	return out;

}

/*im2col_get_pixel(data_im, height, width, channels,im_row, im_col, c_im, pad);
**  ������Ķ�ͨ������im���洢ͼ�����ݣ��л�ȡָ���С��С���ͨ��������Ԫ��ֵ
**  ���룺 im      ���룬�������ݴ��һ��һά���飬�������3ͨ���Ķ�άͼ����ԣ�
**                ÿһͨ�����д洢��ÿһͨ�������в���һ�У�����ͨ�������ٲ���һ��
**        height  ÿһͨ���ĸ߶ȣ�������ͼ��������ĸ߶ȣ���0֮ǰ��
**        width   ÿһͨ���Ŀ�ȣ�������ͼ��Ŀ�ȣ���0֮ǰ��
**        channels ����im��ͨ�����������ɫͼΪ3ͨ����֮��ÿһ�����������ͨ����������һ��������˵ĸ���
**        row     Ҫ��ȡ��Ԫ�����ڵ��У���άͼ��0֮���������
**        col     Ҫ��ȡ��Ԫ�����ڵ��У���άͼ��0֮���������
**        channel Ҫ��ȡ��Ԫ�����ڵ�ͨ��
**        pad     ͼ���������¸���0�ĳ��ȣ��ı߲�0�ĳ���һ����
**  ���أ� float�������ݣ�Ϊim��channelͨ����row-pad�У�col-pad�д���Ԫ��ֵ
**  ע�⣺��im�в�û�д洢��0��Ԫ��ֵ�����height��width����û�в�0ʱ����ͼ��������
**       �ߡ�����row��col���ǲ�0֮��Ԫ�����ڵ����У���ˣ�Ҫ׼ȷ��ȡ��im�е�Ԫ��ֵ��
**       ������Ҫ��ȥpad�Ի�ȡ��im����ʵ��������
*/
float get_pixel(float *im, int inh, int inw,int channel, int row, int col,  int pad)
{
	// ��ȥ��0���ȣ���ȡԪ������ʵͼ���е�������
	row -= pad;
	col -= pad;

	// ���������С��0,���ߴ��ڵ�������ͼ��Ŀ��ߣ��򷵻�0���պ��ǲ�0��Ч����
	if (row < 0 || col < 0 ||
		row >= inh || col >= inh) return 0;
	// im�洢��ͨ����άͼ������ݵĸ�ʽΪ����ͨ�������в���һ�У��ٶ�ͨ�����β���һ�У�
	// ���width*height*channel������λ������ͨ�������λ�ã��ټ���width*row��λ������ָ��ͨ�������У�
	// ������col��λ��������
	return im[col + inw*(row + inh*channel)];
}

//
//	// ���������������ת180�ȵ�����ģ���������
//	im2col_cpu(, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
//
//	gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
//
//}
//
///*
//** im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
//**        ����������ͼ��ͨ����������ͼ��߶ȡ���ȡ�����˳ߴ硢������padding��С���������������
//** ���룺 data_im	����ͼ��
//**		 channels	����ͼ���ͨ���������ڵ�һ�㣬һ������ɫͼ��3ͨ�����м��ͨ����Ϊ��һ�����˸�����
//**		 height 	����ͼ��ĸ߶ȣ��У�
//**		 width		����ͼ��Ŀ�ȣ��У�
//**		 ksize		����˳ߴ�
//**		 stride 	����
//**		 pad		���ܲ�0����
//**		 data_col	�൱�������Ϊ���и�ʽ���ź������ͼ������
//** ˵����ʵ��data_col�е�Ԫ�ظ���Ϊchannels*ksize*ksize*height_col*width_col��
//**      ����channelsΪdata_im��ͨ������ksizeΪ����˴�С��height_col��width_col������ע��
//**      data_col������Ϊchannels*ksize*ksize��=ÿ���а�����ĳ��λ�ô��ľ���˼��������ͨ���ϵ����أ�,
//**      ����������ͼ��ͨ����Ϊ3,����˳ߴ�Ϊ3*3������27�У�ÿ����27��Ԫ�أ���
//**      data_col������Ϊheight_col*width_col����һ������ͼ�ܵ�Ԫ�ظ�������ͬ�ж�Ӧ�������ͼ���ϵĲ�ͬλ�������
//*/
//float** im2col_cpu(float* data_im,
//	int channels, int height, int width,
//	int ksize, int stride, int pad, float* data_col)
//{
//	int c, h, w;
//	// ����ò�����������ͼ��ߴ磨��ʵû�б�Ҫ�ٴμ���ģ���Ϊ�ڹ��������ʱ��make_convolutional_layer()����
//	// �Ѿ�����convolutional_out_width()��convolutional_out_height()������ȡ��������������
//	// �˴�ֱ��ʹ��l.out_h,l.out_w���ɣ���������ֻҪ����ò�����ָ��Ϳ��ԣ�
//	int height_col = (height + 2 * pad - ksize) / stride + 1;     //���ͼ��ĸ߶�
//	int width_col = (width + 2 * pad - ksize) / stride + 1;       //���ͼ��Ŀ��
//	/// ����˴�С��ksize*ksize��һ������˵Ĵ�С��֮���Գ���ͨ����channels������Ϊ����ͼ���ж�ͨ����ÿ��������������ʱ��
//	/// ��ͬʱ��ͬһλ�ô���ͨ����ͼ����о�����㡣����Ϊ��ʵ����һĿ�ģ�����ͨ���ϵľ���˲���һ���Ա���м��㣬��˾����
//	/// ʵ���ϲ����Ƕ�ά�ģ�������ά�ġ��������3ͨ��ͼ�񣬾���˳ߴ�Ϊ3*3���þ���˽�ͬʱ��������ͨ��ͼ���ϣ������������͵�
//	/// ������27��Ԫ�صľ���ˣ�����27��Ԫ�ض��Ƕ�������Ҫѵ���Ĳ����������ڼ���ѵ����������ʱ��һ��Ҫע��ÿһ������˵�ʵ��
//	/// ѵ��������Ҫ��������ͨ������
//	int channels_col = channels * ksize * ksize;��//im2col��ľ�����������3*3*3=27
//
//	//������ź��ͼ������Ϊchannels_col������Ϊheight_col*width_col       
//	// ******������ѭ��֮����߼���ϵ������������ͼ�����ź�ĸ�ʽ��*******     
//	// ��ѭ������Ϊһ������˵ĳߴ�����ѭ��������Ϊ���յõ���data_col��������    
//	for (c = 0; c < channels_col; ++c) {     //�ȵõ�col_data�ĵ�һ�У���c++�õ����е��С�
//		// ������ڵ���ƫ�ƣ��������һ����ά���󣬲����У���������һ��3�У��洢��һά�����У��������������ȡ��Ӧ�ھ�����е��������������
//		// 3*3�ľ���ˣ�3ͨ��������c=0ʱ����Ȼ�ڵ�һ�У���c=5ʱ����Ȼ�ڵ�2�У���c=9ʱ���ڵڶ�ͨ���ϵľ���˵ĵ�һ�У�
//		// ��c=26ʱ���ڵ����У�����ͨ���ϣ�
//		int w_offset = c % ksize;    //����3���������Ǿ���˴�С�ڵ���ƫ��
//
//		// ����˴�С�ڵ���ƫ�ƣ��������һ����ά�ľ������ǰ��У�����������в���һ�У��洢��һά�����еģ�
//		// �������3*3�ľ���ˣ�����3ͨ����ͼ����ôһ������˾���27��Ԫ�أ�ÿ9��Ԫ�ض�Ӧһ��ͨ���ϵľ���ˣ���Ϊһ������
//		// ÿ��cΪ3�ı���������ζ�ž���˻���һ�У�h_offsetȡֵΪ0,1,2����Ӧ3*3������еĵ�1, 2, 3��
//		int h_offset = (c / ksize) % ksize;    //�õ���Ӧһͨ���ϵ�����
//
//		// ͨ��ƫ�ƣ��ھ�����еģ���channels_col�Ƕ�ͨ���ľ���˲���һ��ģ��������3ͨ����3*3����ˣ�ÿ��9��Ԫ�ؾ�Ҫ��һͨ������
//		// ��c=0~8ʱ��c_im=0;��c=9~17ʱ��c_im=1;��c=18~26ʱ��c_im=2
//		int c_im = c / ksize / ksize;�� //����Ŀǰ����ͼ��ĵڼ���ͨ��
//
//		// ��ѭ���������ڸò����ͼ������height_col��˵��data_col�е�ÿһ�д洢��һ������ͼ����������ͼ���ǰ��д洢��data_col�е�ĳ����
//		for (h = 0; h < height_col; ++h) {
//			// ��ѭ�����ڸò����ͼ������width_col��˵�����յõ���data_col����channels_col�У�height_col*width_col��
//			for (w = 0; w < width_col; ++w) {
//				// �������֪������3*3�ľ���ˣ�h_offsetȡֵΪ0,1,2,��h_offset=0ʱ������ȡ�����������˵�һ��Ԫ�ؽ�����������أ�
//				// �������ƣ�����h*stride�ǶԾ���˽�������λ�������������˴�ͼ��(0,0)λ�ÿ�ʼ���������ô���ȿ�ʼ�漰(0,0)~(3,3)
//				// ֮�������ֵ����stride=2����ô����˽���һ������λʱ����һ�еľ�������Ǵ�Ԫ��(2,0)��2Ϊͼ���кţ�0Ϊ�кţ���ʼ
//				int im_row = h_offset + h * stride;
//
//				// ����3*3�ľ���ˣ�w_offsetȡֵҲΪ0,1,2����w_offsetȡ1ʱ������ȡ�������������е�2��Ԫ�ؽ�����������أ�
//				// ʵ�������������ʱ������˶�ͼ������ɨ�������������w*stride����Ϊ��������λ��
//				// ����ǰһ�ξ����ʵ����Ԫ��Ϊ(0,0)����stride=2,��ô�´ξ��Ԫ����ʼ����λ��Ϊ(0,2)��0Ϊ�кţ�2Ϊ�кţ�
//				int im_col = w_offset + w * stride;
//
//				// col_indexΪ���ź�ͼ���е���������������c * height_col * width_col + h * width_col +w�����ǰ��д洢������ͨ���ٲ���һ�У���
//				// ��Ӧ��cͨ����h�У�w�е�Ԫ��
//				int col_index = (c * height_col + h) * width_col + w;
//
//				// im2col_get_pixel������ȡ����ͼ��data_im�е�c_imͨ����im_row,im_col������ֵ����ֵ�����ź��ͼ��
//				// height��widthΪ����ͼ��data_im����ʵ�ߡ���padΪ���ܲ�0�ĳ��ȣ�ע��im_row,im_col�ǲ�0֮������кţ�
//				// ������ʵ����ͼ���е����кţ������Ҫ��ȥpad��ȡ��ʵ�����кţ�
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
//**  ������  A,C������M��B,C������N��ALPGA=1,Ȩ��A��A���С�col_data B��B��������������C��C������
//**	���ܣ���gemm_cpu()�������ã�ʵ�����C = ALPHA * A * B + C ������㣬
//**		 �����CҲ�ǰ��д洢�������в���һ�У�
//**	���룺 A,B,C   �������һά�����ʽ��
//**		  ALPHA   ϵ��
//**		  BETA	  ϵ��
//**		  M 	  A,C������������ת�ã�����A'����������ת�ã����˴�Aδת�ã���ΪA������
//**		  N 	  B,C������������ת�ã�����B'����������ת�ã����˴�Bδת�ã���ΪB������
//**		  K 	  A������������ת�ã�����A'����������ת�ã���B������������ת�ã�����B'����������ת�ã����˴�A,B��δת�ã���ΪA��������B������
//**		  lda	  A������������ת�ã�����A'����������ת�ã����˴�Aδת�ã���ΪA������
//**		  ldb	  B������������ת�ã�����B'����������ת�ã����˴�Bδת�ã���ΪB������
//**		  ldc	  C������
//**	˵��1���˺�������Cʵ�־���˷����㣬�ⲿ�ִ���ģ��Caffe�е�math_functions.cpp�Ĵ���
//**		 �ο����ͣ�http://www.voidcn.com/blog/thy_2014/article/p-6149690.html
//**		 ��Ϊ��ϸ��ע�Ͳμ���gemm_cpu()������ע��
//**	˵��2���˺�����gemm_cpu()�����е��ã��������������֮һ��A,B��������ת��
//**		 ��������gemm_nn()�е�����nn�ֱ��ʾnot transpose�� not transpose
//*/
//
//float** gemm_nn(int M, int N, int K, float ALPHA,
//	float *A, int lda,
//	float *B, int ldb,
//	float *C, int ldc)
//{	// input: ����A[M,K], filter: ����B[K,N],  output: ����C[M,N]
//	int i, j, k;
//#pragma omp parallel for
//	// ��ѭ�����������C(����Ȩ��A)��ÿһ�У�i��ʾA�ĵ�i�У�Ҳ��C�ĵ�i��
//	for (i = 0; i < M; ++i){
//		// ��ѭ������������Ȩ��Aÿһ�е������У�k��ʾȨ��A�ĵ�k�У�ͬʱ��ʾB�ĵ�k��
//		for (k = 0; k < K; ++k){
//			// �ȼ���ALPHA * A��A��ÿ��Ԫ�س���ALPHA��
//			register float A_PART = ALPHA*A[i*lda + k];
//			// ��ѭ��������B�����C���������У�ÿ�δ�ѭ����ϣ�������õ�A*Bһ�еĽ��
//			// j��B�ĵ�j�У�Ҳ��C�ĵ�j��
//			for (j = 0; j < N; ++j){
//				// A�еĵ�i��k����B�е�k��j�ж�Ӧ��ˣ���Ϊһ����ѭ��Ҫ����A*B����֮�����
//				// ��ˣ���������һ����ѭ������û��ֱ�ӳ���B[k*ldb+i]
//				// ÿ����ѭ����ϣ�������A*B���еĲ��ֽ����A�е�i��k��!��B�����е�k��!����Ԫ����˵Ľ��
//				C[i*ldc + j] += A_PART*B[k*ldb + j];
//			}
//		}
//	}
//}





///*
//#define BUFFER_SIZE 2
//�ϲ��ļ�mergeFile (infile1, infile2)
//void mergeFile(const char *fp1, const char *fp2)
//{
//	FILE *fd1, *fd2;           //���������ļ�ָ��
//	float buf[1];
//
//	int rc1, i=0;
//	fd1 = fopen(fp1, "rb");   //��Դ�ļ�
//	fd2 = fopen(fp2, "ab");   //��Ŀ���ļ�
//
//	while ((rc1 = fread(buf, sizeof(float), 1, fd1)) != 0)    //��ȡԴ�ļ��е�һ��float���ݣ����Ƶ�buf
//	{
//		i++;
//		fwrite(buf, sizeof(float), rc1, fd2);                 //�����Ƶ�buf�е����ݣ�������Ŀ���ļ���
//	}
//	printf("%d\n",i);
//	
//	Sleep(0.1);              //�ȴ�����
//	fclose(fd1);             //�ر�Դ�ļ�
//	fclose(fd2);			 //�ر�Ŀ���ļ�
//
//}
//*/
///*
//	�Ծ��������cnn���г�ʼ��,
//	����˵Ĵ�СΪ3x3,stride=1,same padding 
//	�ػ��˴�С2x2��stride=2
//	���������ĳ�ʼ����Ҫ�����˴��������ݵ��ڴ�ռ估Ȩ�ص������ֵ�����սṹ����ռ�Ϳ�����
//	cnnsetup(cnn,inputSize,outSize)��������CNN��������ṹ�塢��������Ŀ��ߡ����������ά��
//
//void cnnsetup(CNN* cnn,nSize inputSize,int outputSize)
//{
//	cnn->layerNum=6;    //������ܲ���Ϊ5
//
//	nSize inSize;         //�����ߡ���ֵ�Ľṹ��
//	int mapSize=3;              //����˵Ĵ�С3x3
//
//	c1�����ĳ�ʼ�������س�ʼ���õľ����ṹ��
//	���������������ߡ�����˳ߴ硢����ͨ���������ͨ����(stride=1,same padding)
//	inSize.c=inputSize.c;       //C1������Ŀ�
//	inSize.r=inputSize.r;       //C1���������
//	cnn->C1=initCovLayer(inSize.c,inSize.r,3,1,16);								 //��ʼ��C1��
//
//	s2�ػ���ĳ�ʼ�������س�ʼ���õĳػ���
//	���������������ߡ��ػ��˴�С������ͨ���������ͨ�������ػ�����
//	inSize.c=inSize.c;			 //C1������Ŀ�=S2������Ŀ�
//	inSize.r=inSize.r;			 //C1������ĸ�=S2������Ŀ�
//	cnn->S2=initPoolLayer(inSize.c,inSize.r,2,16,16,MaxPool);					//��ʼ��S2��
//
//	c3�����ĳ�ʼ�������س�ʼ���õľ����ṹ��
//	        ���������������ߡ�����˳ߴ硢����ͨ���������ͨ����
//	inSize.c=inSize.c/2;		 //S2��������=C3������Ŀ�=n/2
//	inSize.r=inSize.r/2;		 //S2��������=C3���������=n/2
//	cnn->C3=initCovLayer(inSize.c,inSize.r,3,16,32);							//��ʼ��C3��
//
//	s4�ػ���ĳ�ʼ�������س�ʼ���õĳػ���
//	inSize.c=inSize.c;			 //S4������Ŀ�
//	inSize.r=inSize.r;			 //S4���������
//	cnn->S4=initPoolLayer(inSize.c,inSize.r,2,32,32,MaxPool);					 //��ʼ��S4��
//
//	inSize.c=inSize.c/2;         //05������Ŀ�=n/2
//	inSize.r=inSize.r/2;         //05���������=n/2
//	05���ȫ���Ӳ�ĳ�ʼ��
//	������	     ������Ԫ�����������Ԫ����
//	cnn->O5=initOutLayer(inSize.c*inSize.r*32,128);								 //��ʼ��05��
//
//	05���ȫ���Ӳ�ĳ�ʼ��
//	cnn->O6=initOutLayer(128, outputSize);									 //��ʼ��06��
//	cnn->e=(float*)calloc(cnn->O5->outputNum,sizeof(float));					 //ѵ�����
//}
//*/
///*
//cnn��ĳ�ʼ��
//cnn->C1=initCovLayer(inSize.c,inSize.r,3,1,16);    ���1������6				  //��ʼ��CU��
//cnn->C3=initCovLayer(inSize.c,inSize.r,3,16,32);	���6������12				 //��ʼ��C3��
//7������         ���������ߡ�����˳ߴ硢����ͨ���������ͨ������padding��С��������С
//CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels,int pad,int stride)
//{
//	CovLayer* covL=(CovLayer*)malloc(sizeof(CovLayer));    //�½������Ľṹ��
//
//	
//	covL->inputHeight=inputHeight;                    //�������������ͼ���
//	covL->inputWidth=inputWidth;					  //�������������ͼ���
//	covL->mapSize=mapSize;							  //��������ľ���˴�С
//	covL->inChannels=inChannels;					  //�������������ͨ����
//	covL->outChannels=outChannels;                    //������������ͨ����
//	covL->pad = pad;								  //�������������padding��С
//	covL->stride = stride;							  //��������Ĳ�����С
//
//	covL->isFullConnect=true;						 // Ĭ��Ϊȫ����
//
//	 ����Ȩ�ص��ڴ�ռ䣨--4άfloat�ͣ������д洢
//	int i,j,c,r;
//	srand((unsigned)time(NULL)); //������������ӵĺ����������ڵ�ϵͳʱ����Ϊ����������������������
//								 rand������һ������������ĺ����������������ʹ��
//
//	covL->mapData=(float****)malloc(outChannels*sizeof(float***));  //��������ȿռ�
//	for(i=0;i<outChannels;i++){   //����n������ռ�
//		covL->mapData[i]=(float***)malloc(outChannels*sizeof(float**));
//		for(j=0;j<inChannels;j++){  //һ������ѭ������һ������ͼ�Ĵ�С
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
//	����ƫ�õ��ڴ�ռ䣨--1άfloat�ͣ�
//	covL->basicData=(float*)calloc(outChannels,sizeof(float));
//
//	�����������ͼ���ڴ�ռ䣨����������ĺ�δ����������ģ�--3ά
//	���������ĳߴ��С
//	int outW=inputWidth;
//	int outH=inputHeight;
//	
//	covL->v=(float***)malloc(outChannels*sizeof(float**));  //Ϊ����ǰ��ֵ����ռ䣨3άfloat�ͣ�
//	covL->y=(float***)malloc(outChannels*sizeof(float**));  //Ϊ������ֵ����ռ䣨3άfloat�ͣ�
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
//	return covL;      //���س�ʼ���õľ����
//}
//*/
//
///*
//cnn->S2=initPoolLayer(inSize.c,inSize.r,2,16,16,MaxPool)     //��ʼ��S2��
//cnn->S4=initPoolLayer(inSize.c,inSize.r,2,32,32,MaxPool);  //��ʼ��S4��
//�ػ���ĳ�ʼ����
//6������   ���������ߡ��ػ��˴�С��������������ػ�����
//PoolLayer* initPoolLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels,int poolType)
//{
//	PoolLayer* poolL=(PoolLayer*)malloc(sizeof(PoolLayer));    //�½��ػ���Ľṹ��
//
//	����ػ��������ߡ�������ػ��˴�С������ͨ���������ͨ�������ػ�����
//	poolL->inputHeight=inputHeight;                  //����ػ���������
//	poolL->inputWidth=inputWidth;                    //����ػ���������
//	poolL->mapSize=mapSize;                          //����ػ��˴�С
//	poolL->inChannels=inChannels;                    //����ػ��������ͨ����
//	poolL->outChannels=outChannels;                  //����ػ�������ͨ����
//	poolL->poolType=poolType;                        //�ػ�����
//
//	����ƫ�õ��ڴ�ռ䣨--1άfloat�ͣ�
//	poolL->basicData=(float*)calloc(outChannels,sizeof(float));
//	
//	�����������ͼ����ά�ڴ�ռ�
//	int outW=inputWidth/mapSize;  
//	int outH=inputHeight/mapSize;
//	int j,r;
//	poolL->d=(float***)malloc(outChannels*sizeof(float**));    // ����ľֲ��ݶ�,��ֵ
//	poolL->y=(float***)malloc(outChannels*sizeof(float**));    // ������������Ԫ�������3άfloat�ͣ�,�޼����������ռ�
//	for(j=0;j<outChannels;j++){
//		poolL->d[j]=(float**)malloc(outH*sizeof(float*));      
//		poolL->y[j]=(float**)malloc(outH*sizeof(float*)); 
//		for(r=0;r<outH;r++){
//			poolL->d[j][r]=(float*)calloc(outW,sizeof(float));
//			poolL->y[j][r]=(float*)calloc(outW,sizeof(float));
//		}
//	}
//
//	return poolL;   //���س�ʼ���õĳػ���
//}
//*/
///*
//���ȫ���Ӳ�ĳ�ʼ��
// cnn->O5 = initOutLayer(inSize.c*inSize.r * 12, outputSize);    //��ʼ��05��
//������				������Ԫ�����������Ԫ����
//OutLayer* initOutLayer(int inputNum,int outputNum)
//{
//	OutLayer* outL=(OutLayer*)malloc(sizeof(OutLayer));      //�½������Ľṹ��
//
//	�������ȫ���Ӳ������������Ŀ�����������Ŀ��ƫ�á�
//	outL->inputNum=inputNum;                     //�������ȫ���Ӳ������������Ŀ          
//	outL->outputNum=outputNum;                   //�������ȫ���Ӳ�����������Ŀ
//
//	����ƫ�õķ���ռ䣨float�ͣ�
//	outL->basicData=(float*)calloc(outputNum,sizeof(float));
//
//	 ����ȫ���Ӳ�����Ŀռ�ռ䣨1άfloat�ͣ�
//	outL->d=(float*)calloc(outputNum,sizeof(float));
//	outL->v=(float*)calloc(outputNum,sizeof(float));   // Ϊ���뼤���������ֵ����ռ�
//	outL->y=(float*)calloc(outputNum,sizeof(float));   // Ϊ���������Ԫ���������ռ�
//
//	 ����ȫ���Ӳ�Ȩ�صĳ�ʼ������ռ䣨2άfloat�ͣ�
//	outL->wData=(float**)malloc(outputNum*sizeof(float*)); // �����У������
//	int i,j;
//	srand((unsigned)time(NULL));
//	for(i=0;i<inputNum;i++){
//		outL->wData[i]=(float*)malloc(inputNum*sizeof(float));  // Ϊȫ���Ӳ�Ȩ�صĳ�ʼ������ռ䣨float�ͣ�
//		for(j=0;j<outputNum;j++){
//			float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2; // ����һ��-1��1�������
//			outL->wData[i][j]=randnum*sqrt((float)6.0/(float)(inputNum+outputNum));
//		}
//	}
//
//	outL->isFullConnect=true; 
//
//	return outL;  //���س�ʼ���õ������
//}
//
//*/
//(vecmaxIndex(cnn->O5->y,cnn->O5->outputNum)
//int vecmaxIndex(float* vec, int veclength)// ������������������
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
// ����cnn����
// incorrectRatio=cnntest(cnn,testImg,testLabel,testNum);
//������         CNN����ṹ�塢�������ݵĽṹ�塢���Ա�ǩ�Ľṹ�塢������Ŀ��1��
//float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum)
//{
//	int n=0,outindex;
//	int incorrectnum=0;  //����Ԥ�����Ŀ
//	for(n=0;n<testNum;n++){        //����������Ŀ
//		cnnff(cnn,inputData->ImgPtr[n].ImgData);    //�Ե�n��ͼƬ���ݽ���ǰ�򴫲�
//		/*
//		if(vecmaxIndex(cnn->O5->y,cnn->O5->outputNum)!=vecmaxIndex(outputData->LabelPtr[n].LabelData,cnn->O5->outputNum))
//			incorrectnum++;
//		*/
//		outindex = vecmaxIndex(cnn->O6->y, cnn->O6->outputNum);  //������һ���������
//		printf("����Ľ����%d", outindex);
//		cnnclear(cnn);
//		printf("testing: %d\n", n);    //��ӡ���Ե��ִ�
//	}
//	return (float)incorrectnum/(float)testNum;
//}
//
//
///*
// ����ѵ���õ�cnn��Ȩ������filename�ļ���cnn������
//void importcnn(CNN* cnn,   char* filename)
//{	
//	��Ȩ���ļ�
//	char* filedir = "D:\\FPGA-AI\\Lenet_MNIST\\DeepLearningC-master\\CNN\\CNNdata";
//	const char* allweigth_filename = filename;   //��Ȩ���ļ�·��תΪ��һ��ָ���ַ���������ָ��
//	const char* c1_filename = "D:\\FPGA-AI\\Lenet_MNIST\\DeepLearningC-master\\CNN\\weight_bin\\c1w.bin";
//	FILE  *fp=NULL;									//����һ���ļ�ָ��
//	fp = fopen(allweigth_filename, "rb");          //��ѵ���õ�����Ȩ���ļ�allweigth_file.bin
//	if (fp == NULL)
//		printf("write file failed2\n");
//	else printf("read all_weight ok!\n");
//
//	 ��ȡC1�ľ��Ȩ�أ�4άfloat�����飩
//	int i,j,c,r;
//	
//	for(i=0;i<cnn->C1->outChannels;i++)
//		for(j=0;j<cnn->C1->inChannels;j++)
//			for(r=0;r<cnn->C1->mapSize;r++)
//				for(c=0;c<cnn->C1->mapSize;c++){
//					float* in=(float*)malloc(sizeof(float));   //����һ��float��Ȩ�ص��ڴ�ռ�
//					fread(in, sizeof(float), 1, fp);           //��Ȩ���ļ��ж�ȡһ��float�ֽڵ�Ȩ�ص�inָ��
//					cnn->C1->mapData[i][j][r][c]=*in;		   //����ȡ��һ��Ȩֵin��ֵ�������Ȩ��mapData
//				}
//	printf("import c1 weight ok!--");
//	fclose(fp);
//
//	
//	��ȡ C1��ƫ�ã�1άfloat�����飩
//	for(i=0;i<cnn->C1->outChannels;i++)
//		fread(&cnn->C1->basicData[i], sizeof(float), 1, fp);
//	printf("import c1b ok!--");
//	
//
//	
//	 ��ȡC3����ľ��Ȩ�أ�4άfloat�����飩
//	for(i=0;i<cnn->C3->outChannels;i++)
//		for(j=0;j<cnn->C3->inChannels;j++)
//			for(r=0;r<cnn->C3->mapSize;r++)
//				for(c=0;c<cnn->C3->mapSize;c++)
//				fread(&cnn->C3->mapData[i][j][r][c],sizeof(float),1,fp);
//	printf("import c3 weight ok!--");
//	��ȡC3��ƫ�ã�1άfloat�����飩
//	for(i=0;i<cnn->C3->outChannels;i++)
//		fread(&cnn->C3->basicData[i],sizeof(float),1,fp);
//	printf("import c3b ok!--");
//
//	 ��ȡf5�����ľ��Ȩ�أ�2άfloat�����飩
//	for(i=0;i<cnn->O5->outputNum;i++)
//		for(j=0;j<cnn->O5->inputNum;j++)
//			fread(&cnn->O5->wData[i][j],sizeof(float),1,fp);
//	printf("import f5w ok!--");
//	��ȡO5������ƫ�ã�1άfloat�����飩
//	for(i=0;i<cnn->O5->outputNum;i++)
//		fread(&cnn->O5->basicData[i],sizeof(float),1,fp);
//	printf("import f5b ok!--");
//
//	 ��ȡf6�����ľ��Ȩ�أ�2άfloat�����飩
//	for (i = 0; i<cnn->O6->outputNum; i++)
//		for (j = 0; j<cnn->O6->inputNum; j++)
//			fread(&cnn->O6->wData[i][j], sizeof(float), 1, fp);
//	printf("import f6w ok!--");
//	��ȡO6������ƫ�ã�1άfloat�����飩
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
// ����� input�����ݣ�inputNum˵��������Ŀ��bas����ƫ��
//float activation_Sigma(float input,float bas) // sigma�����
//{
//	float temp=input+bas;
//	return (float)1.0/((float)(1.0+exp(-temp)));
//}
//
//
//ƽ���ػ�
//void avgPooling(float** output,nSize outputSize,float** input,nSize inputSize,int mapSize) // ��ƽ��ֵ
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
// ����ȫ�����������ǰ�򴫲�
//float vecMulti(float* vec1,float* vec2,int vecL)// ���������
//{
//	int i;
//	float m=0;
//	for(i=0;i<vecL;i++)
//		m=m+vec1[i]*vec2[i];
//	return m;
//}
//
//ȫ���ӵ�ǰ�򴫲�
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
//float sigma_derivation(float y){ // Logic��������Ա���΢��
//	return y*(1-y); // ����y��ָ��������������ֵ���������Ա���
//}
//
//
//
//
//
//������м��������㣬ֻ����Ȩ������
//void cnnclear(CNN* cnn)
//{
//	 ����Ԫ�Ĳ����������
//	int j,c,r;
//	 C1����
//	for(j=0;j<cnn->C1->outChannels;j++){
//		for(r=0;r<cnn->S2->inputHeight;r++){
//			for(c=0;c<cnn->S2->inputWidth;c++){
//				cnn->C1->d[j][r][c]=(float)0.0;
//				cnn->C1->v[j][r][c]=(float)0.0;
//				cnn->C1->y[j][r][c]=(float)0.0;
//			}
//		}
//	}
//	 S2����
//	for(j=0;j<cnn->S2->outChannels;j++){
//		for(r=0;r<cnn->C3->inputHeight;r++){
//			for(c=0;c<cnn->C3->inputWidth;c++){
//				cnn->S2->d[j][r][c]=(float)0.0;
//				cnn->S2->y[j][r][c]=(float)0.0;
//			}
//		}
//	}
//	 C3����
//	for(j=0;j<cnn->C3->outChannels;j++){
//		for(r=0;r<cnn->S4->inputHeight;r++){
//			for(c=0;c<cnn->S4->inputWidth;c++){
//				cnn->C3->d[j][r][c]=(float)0.0;
//				cnn->C3->v[j][r][c]=(float)0.0;
//				cnn->C3->y[j][r][c]=(float)0.0;
//			}
//		}
//	}
//	 S4����
//	for(j=0;j<cnn->S4->outChannels;j++){
//		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
//			for(c=0;c<cnn->S4->inputWidth/cnn->S4->mapSize;c++){
//				cnn->S4->d[j][r][c]=(float)0.0;
//				cnn->S4->y[j][r][c]=(float)0.0;
//			}
//		}
//	}
//	 O5���
//	for(j=0;j<cnn->O5->outputNum;j++){
//		cnn->O5->d[j]=(float)0.0;
//		cnn->O5->v[j]=(float)0.0;
//		cnn->O5->y[j]=(float)0.0;
//	}
//}
//
// �������ڲ��Եĺ���
//void savecnndata(CNN* cnn,const char* filename,float** inputdata) // ����CNN�����е��������
//{
//	FILE  *fp=NULL;
//	fp=fopen(filename,"wb");
//	if(fp==NULL)
//		printf("write file failed3\n");
//
//	 C1������
//	int i,j,r;
//	 C1����
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
//	 S2����
//	for(j=0;j<cnn->S2->outChannels;j++){
//		for(r=0;r<cnn->C3->inputHeight;r++){
//			fwrite(cnn->S2->d[j][r],sizeof(float),cnn->C3->inputWidth,fp);
//		}
//		for(r=0;r<cnn->C3->inputHeight;r++){
//			fwrite(cnn->S2->y[j][r],sizeof(float),cnn->C3->inputWidth,fp);
//		}
//	}
//	 C3����
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
//	 S4����
//	for(j=0;j<cnn->S4->outChannels;j++){
//		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
//			fwrite(cnn->S4->d[j][r],sizeof(float),cnn->S4->inputWidth/cnn->S4->mapSize,fp);
//		}
//		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
//			fwrite(cnn->S4->y[j][r],sizeof(float),cnn->S4->inputWidth/cnn->S4->mapSize,fp);
//		}
//	}
//
//	 O5�����
//	for(i=0;i<cnn->O5->outputNum;i++)
//		fwrite(cnn->O5->wData[i],sizeof(float),cnn->O5->inputNum,fp);
//	fwrite(cnn->O5->basicData,sizeof(float),cnn->O5->outputNum,fp);
//	fwrite(cnn->O5->v,sizeof(float),cnn->O5->outputNum,fp);
//	fwrite(cnn->O5->d,sizeof(float),cnn->O5->outputNum,fp);
//	fwrite(cnn->O5->y,sizeof(float),cnn->O5->outputNum,fp);
//
//	fclose(fp);
//}
