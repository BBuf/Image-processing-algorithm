#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
//所有代码针对灰度图，RGB分为3个通道处理
//串行求两个向量的距离
float getDistanceCPU(float *a, float *b, const int len){
	float dis = 0.0f;
	for (int i = 0; i < len; i++) {
		dis += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrtf(dis);
}

//x86循环展开BX次优化求2个向量的距离
float getDistancex86(float *a, float *b, const int len, const int BX) {
	float *result = new float[BX];
	for (int i = 0; i < BX; i++) result[i] = 0.0f;
	float *temp = new float[BX];
	int i;
	for (i = 0; i + BX < len; i += BX) {
		for (int ii = 0; ii < BX; ii++) {
			temp[ii] = (a[i + ii] - b[i + ii]);
			result[ii] += temp[ii] * temp[ii];
		}
	}
	for (int ii = 1; ii < BX; ii++) {
		result[0] += result[ii];
	}
	for (; i < len; i++) {
		result[0] += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrtf(result[0]);
}

//AVX指令加速两向量的距离运算，性能最多得到了3倍提升
float inline reduceM128(__m128 r) {
	float f[4];
	_mm_store_ps(f, r);
	return (f[0] + f[1]) + (f[2] + f[3]);
}
float inline reduceM256(__m256 r) {
	__m128 h = _mm256_extractf128_ps(r, 1);
	__m128 l = _mm256_extractf128_ps(r, 0);
	h = _mm_add_ps(h, l);
	return reduceM128(h);
}
//_mm256_fmadd_ps使用乘加指令一次计算完一次乘法和一次加法，可以减少指令数量，提高吞吐量
float dotWithAVX(float *a, float *b, const int len) {
	int step = len / 8;
	__m256* one = (__m256*)a;
	__m256* two = (__m256*)b;
	__m256 result = _mm256_setzero_ps();
	__m256 temp;
	for (int i = 0; i < step; i++) {
		temp = _mm256_sub_ps(one[i], two[i]);
		result = _mm256_fmadd_ps(temp, temp, result);
	}
	float r = reduceM256(result);
	for (int i = 8 * step; i < len; i++) {
		r += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrtf(r);
}

//AVX指令+循环展开加速
float dotWithAVXUnroll(float *a, float *b, const int len, const int BX) {
	int step = len / (8 * BX);
	__m256* one = (__m256*)a;
	__m256* two = (__m256*)b;
	__m256 *result = new __m256[BX];
	for (int i = 0; i < BX; i++) {
		result[i] = _mm256_setzero_ps();
	}
	for (int i = 0; i < step; i++) {
		for (int j = 0; j < BX; j++) {
			__m256 temp = _mm256_sub_ps(one[j + BX * i], two[j + BX * i]);
			result[j] = _mm256_fmadd_ps(temp, temp, result[j]);
		}
	}
	for (int j = 1; j < BX; j++) {
		result[0] = _mm256_add_ps(result[0], result[j]);
	}
	float r = reduceM256(result[0]);
	for (int i = (8 * BX) * step; i < len; i++) {
		r += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrtf(r);
}

int main() {
	const int len = 10;
	float a[len] = { 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 7.0, 8.5, 9.8, 0.4 };
	float b[len] = { 0.4, 0.5, 0.6, 1.8, 2.6, 1.4, 1.7, 1.0, 2.0, 4.0 };
	//串行
	float ans = getDistanceCPU(a, b, len);
	printf("%.5f\n", ans);
	ans = getDistancex86(a, b, len, 4);
	printf("%.5f\n", ans);
	ans = dotWithAVX(a, b, len);
	printf("%.5f\n", ans);
	ans = dotWithAVXUnroll(a, b, len, 4);
	printf("%.5f\n", ans);
	system("pause");
	return 0;
}