#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "mat.h"
#include "cnn.h"

///*
///*
//��������
//����������Ȩ�أ������˴�С������ͼ��ָ�룬�����������ߣ�����ͨ������������padding��
//*/
//float** conv(CovLayer* covlayer, float** inputData)
//{
//
//	// ����������������ת180�ȵ�����ģ���������
//	im2col_cpu(, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
//
//	gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
//
//}
//
///*
//** im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
//**        ����������ͼ��ͨ����������ͼ��߶ȡ����ȡ������˳ߴ硢������padding��С���������������
//** ���룺 data_im	����ͼ��
//**		 channels	����ͼ���ͨ���������ڵ�һ�㣬һ������ɫͼ��3ͨ�����м��ͨ����Ϊ��һ������˸�����
//**		 height 	����ͼ��ĸ߶ȣ��У�
//**		 width		����ͼ��Ŀ��ȣ��У�
//**		 ksize		�����˳ߴ�
//**		 stride 	����
//**		 pad		���ܲ�0����
//**		 data_col	�൱�������Ϊ���и�ʽ���ź������ͼ������
//** ˵����ʵ��data_col�е�Ԫ�ظ���Ϊchannels*ksize*ksize*height_col*width_col��
//**      ����channelsΪdata_im��ͨ������ksizeΪ�����˴�С��height_col��width_col������ע��
//**      data_col������Ϊchannels*ksize*ksize��=ÿ���а�����ĳ��λ�ô��ľ����˼��������ͨ���ϵ����أ�,
//**      ����������ͼ��ͨ����Ϊ3,�����˳ߴ�Ϊ3*3������27�У�ÿ����27��Ԫ�أ���
//**      data_col������Ϊheight_col*width_col����һ������ͼ�ܵ�Ԫ�ظ�������ͬ�ж�Ӧ��������ͼ���ϵĲ�ͬλ��������
//*/
//float** im2col_cpu(float* data_im,
//	int channels, int height, int width,
//	int ksize, int stride, int pad, float* data_col)
//{
//	int c, h, w;
//	// ����ò�����������ͼ��ߴ磨��ʵû�б�Ҫ�ٴμ���ģ���Ϊ�ڹ���������ʱ��make_convolutional_layer()����
//	// �Ѿ�����convolutional_out_width()��convolutional_out_height()������ȡ��������������
//	// �˴�ֱ��ʹ��l.out_h,l.out_w���ɣ���������ֻҪ����ò�����ָ��Ϳ��ԣ�
//	int height_col = (height + 2 * pad - ksize) / stride + 1;     //���ͼ��ĸ߶�
//	int width_col = (width + 2 * pad - ksize) / stride + 1;       //���ͼ��Ŀ���
//	/// �����˴�С��ksize*ksize��һ�������˵Ĵ�С��֮���Գ���ͨ����channels������Ϊ����ͼ���ж�ͨ����ÿ����������������ʱ��
//	/// ��ͬʱ��ͬһλ�ô���ͨ����ͼ����о������㡣����Ϊ��ʵ����һĿ�ģ�����ͨ���ϵľ����˲���һ���Ա���м��㣬��˾�����
//	/// ʵ���ϲ����Ƕ�ά�ģ�������ά�ġ��������3ͨ��ͼ�񣬾����˳ߴ�Ϊ3*3���þ����˽�ͬʱ��������ͨ��ͼ���ϣ������������͵�
//	/// ������27��Ԫ�صľ����ˣ�����27��Ԫ�ض��Ƕ�������Ҫѵ���Ĳ����������ڼ���ѵ����������ʱ��һ��Ҫע��ÿһ�������˵�ʵ��
//	/// ѵ��������Ҫ��������ͨ������
//	int channels_col = channels * ksize * ksize;��//im2col��ľ�����������3*3*3=27
//
//	//������ź��ͼ������Ϊchannels_col������Ϊheight_col*width_col       
//	// ******������ѭ��֮����߼���ϵ������������ͼ�����ź�ĸ�ʽ��*******     
//	// ��ѭ������Ϊһ�������˵ĳߴ�����ѭ��������Ϊ���յõ���data_col��������    
//	for (c = 0; c < channels_col; ++c) {     //�ȵõ�col_data�ĵ�һ�У���c++�õ����е��С�
//		// �������ڵ���ƫ�ƣ���������һ����ά���󣬲����У���������һ��3�У��洢��һά�����У��������������ȡ��Ӧ�ھ������е��������������
//		// 3*3�ľ����ˣ�3ͨ��������c=0ʱ����Ȼ�ڵ�һ�У���c=5ʱ����Ȼ�ڵ�2�У���c=9ʱ���ڵڶ�ͨ���ϵľ����˵ĵ�һ�У�
//		// ��c=26ʱ���ڵ����У�����ͨ���ϣ�
//		int w_offset = c % ksize;    //����3���������Ǿ����˴�С�ڵ���ƫ��
//
//		// �����˴�С�ڵ���ƫ�ƣ���������һ����ά�ľ������ǰ��У������������в���һ�У��洢��һά�����еģ�
//		// �������3*3�ľ����ˣ�����3ͨ����ͼ����ôһ�������˾���27��Ԫ�أ�ÿ9��Ԫ�ض�Ӧһ��ͨ���ϵľ����ˣ���Ϊһ������
//		// ÿ��cΪ3�ı���������ζ�ž����˻���һ�У�h_offsetȡֵΪ0,1,2����Ӧ3*3�������еĵ�1, 2, 3��
//		int h_offset = (c / ksize) % ksize;    //�õ���Ӧһͨ���ϵ�����
//
//		// ͨ��ƫ�ƣ��ھ������еģ���channels_col�Ƕ�ͨ���ľ����˲���һ��ģ��������3ͨ����3*3�����ˣ�ÿ��9��Ԫ�ؾ�Ҫ��һͨ������
//		// ��c=0~8ʱ��c_im=0;��c=9~17ʱ��c_im=1;��c=18~26ʱ��c_im=2
//		int c_im = c / ksize / ksize;�� //����Ŀǰ����ͼ��ĵڼ���ͨ��
//
//		// ��ѭ���������ڸò����ͼ������height_col��˵��data_col�е�ÿһ�д洢��һ������ͼ����������ͼ���ǰ��д洢��data_col�е�ĳ����
//		for (h = 0; h < height_col; ++h) {
//			// ��ѭ�����ڸò����ͼ������width_col��˵�����յõ���data_col����channels_col�У�height_col*width_col��
//			for (w = 0; w < width_col; ++w) {
//				// �������֪������3*3�ľ����ˣ�h_offsetȡֵΪ0,1,2,��h_offset=0ʱ������ȡ������������˵�һ��Ԫ�ؽ�����������أ�
//				// �������ƣ�����h*stride�ǶԾ����˽�������λ��������������˴�ͼ��(0,0)λ�ÿ�ʼ����������ô���ȿ�ʼ�漰(0,0)~(3,3)
//				// ֮�������ֵ����stride=2����ô�����˽���һ������λʱ����һ�еľ��������Ǵ�Ԫ��(2,0)��2Ϊͼ���кţ�0Ϊ�кţ���ʼ
//				int im_row = h_offset + h * stride;
//
//				// ����3*3�ľ����ˣ�w_offsetȡֵҲΪ0,1,2����w_offsetȡ1ʱ������ȡ��������������е�2��Ԫ�ؽ�����������أ�
//				// ʵ��������������ʱ�������˶�ͼ������ɨ��������������w*stride����Ϊ��������λ��
//				// ����ǰһ�ξ�����ʵ����Ԫ��Ϊ(0,0)����stride=2,��ô�´ξ���Ԫ����ʼ����λ��Ϊ(0,2)��0Ϊ�кţ�2Ϊ�кţ�
//				int im_col = w_offset + w * stride;
//
//				// col_indexΪ���ź�ͼ���е���������������c * height_col * width_col + h * width_col +w�����ǰ��д洢������ͨ���ٲ���һ�У���
//				// ��Ӧ��cͨ����h�У�w�е�Ԫ��
//				int col_index = (c * height_col + h) * width_col + w;
//
//				// im2col_get_pixel������ȡ����ͼ��data_im�е�c_imͨ����im_row,im_col������ֵ����ֵ�����ź��ͼ��
//				// height��widthΪ����ͼ��data_im����ʵ�ߡ�����padΪ���ܲ�0�ĳ��ȣ�ע��im_row,im_col�ǲ�0֮������кţ�
//				// ������ʵ����ͼ���е����кţ������Ҫ��ȥpad��ȡ��ʵ�����кţ�
//				data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
//					im_row, im_col, c_im, pad);
//			}
//		}
//	}
//
//}
///*im2col_get_pixel(data_im, height, width, channels,im_row, im_col, c_im, pad);
//**  ������Ķ�ͨ������im���洢ͼ�����ݣ��л�ȡָ���С��С���ͨ��������Ԫ��ֵ
//**  ���룺 im      ���룬�������ݴ��һ��һά���飬�������3ͨ���Ķ�άͼ����ԣ�
//**                ÿһͨ�����д洢��ÿһͨ�������в���һ�У�����ͨ�������ٲ���һ��
//**        height  ÿһͨ���ĸ߶ȣ�������ͼ��������ĸ߶ȣ���0֮ǰ��
//**        width   ÿһͨ���Ŀ��ȣ�������ͼ��Ŀ��ȣ���0֮ǰ��
//**        channels ����im��ͨ�����������ɫͼΪ3ͨ����֮��ÿһ������������ͨ����������һ����������˵ĸ���
//**        row     Ҫ��ȡ��Ԫ�����ڵ��У���άͼ��0֮���������
//**        col     Ҫ��ȡ��Ԫ�����ڵ��У���άͼ��0֮���������
//**        channel Ҫ��ȡ��Ԫ�����ڵ�ͨ��
//**        pad     ͼ���������¸���0�ĳ��ȣ��ı߲�0�ĳ���һ����
//**  ���أ� float�������ݣ�Ϊim��channelͨ����row-pad�У�col-pad�д���Ԫ��ֵ
//**  ע�⣺��im�в�û�д洢��0��Ԫ��ֵ�����height��width����û�в�0ʱ����ͼ��������
//**       �ߡ�������row��col���ǲ�0֮��Ԫ�����ڵ����У���ˣ�Ҫ׼ȷ��ȡ��im�е�Ԫ��ֵ��
//**       ������Ҫ��ȥpad�Ի�ȡ��im����ʵ��������
//*/
//float im2col_get_pixel(float *im, int height, int width, int channels, int row, int col, int channel, int pad)
//{
//	// ��ȥ��0���ȣ���ȡԪ����ʵ��������
//	row -= pad;
//	col -= pad;
//
//	// ���������С��0,�򷵻�0���պ��ǲ�0��Ч����
//	if (row < 0 || col < 0 ||
//		row >= height || col >= width) return 0;
//	// im�洢��ͨ����άͼ������ݵĸ�ʽΪ����ͨ�������в���һ�У��ٶ�ͨ�����β���һ�У�
//	// ���width*height*channel������λ������ͨ�������λ�ã�����width*row��λ��
//	// ����ָ��ͨ�������У��ټ���col��λ��������
//	return im[col + width*(row + height*channel)];
//}
//
//
///*	gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
//**  ������  A,C������M��B,C������N��ALPGA=1,Ȩ��A��A���С�col_data B��B���������������C��C������
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




