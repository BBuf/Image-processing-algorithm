#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
//所有代码针对灰度图，RGB分为3个通道处理
//中值滤波串行代码
void meanFilter(int height, int width, unsigned char * __restrict src, unsigned char * __restrict dst) {
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			unsigned char a[9];
			a[0] = src[i * width + j];
			a[1] = src[i * width + j + 1];
			a[2] = src[i * width + j - 1];

			a[3] = src[(i + 1) * width + j];
			a[4] = src[(i + 1) * width + j + 1];
			a[5] = src[(i + 1) * width + j - 1];

			a[6] = src[(i - 1) * width + j];
			a[7] = src[(i - 1) * width + j + 1];
			a[8] = src[(i - 1) * width + j - 1];
			for (int ii = 0; ii < 5; ii++) {
				for (int jj = ii + 1; jj < 9; jj++) {
					if (a[ii] > a[jj]) {
						unsigned char temp = a[ii];
						a[ii] = a[jj];
						a[jj] = temp;
					}
				}
			}
			dst[i * width + j] = a[4];
		}
	}
	for (int i = 0; i < width; i++) {
		dst[i] = src[i];
		dst[(height - 1) * width + i] = src[(height - 1) * width + i];
	}
	for (int i = 0; i < height; i++) {
		dst[i * width] = src[i * width];
		dst[i * width + width - 1] = src[i * width + width - 1];
	}
}
Mat speed_MedianFilter(Mat src) {
	int row = src.rows;
	int col = src.cols;
	unsigned char * data = (unsigned char *)src.data;
	unsigned char *dst = new unsigned char[row * col];
	//for (int i = 0; i < row * col; i++) {
	//	printf("%d\n", data[i]);
	//}
	meanFilter(row, col, data, dst);
	Mat res(row, col, CV_8UC1, dst);
	return res;
}

//使用AVX指令集优化中值滤波，相比串行，想你那好提升约60倍

void medianFilterAVX(int height, int width, unsigned char* __restrict src, unsigned char* __restrict dst) {
	for (int i = 1; i < height - 1; i++) {
		int j;
		for (j = 1; j < width - 1 - 32; j += 32) {
			__m256i a[9];
			a[0] = _mm256_loadu_si256((__m256i*)(src + i * width + j));
			a[1] = _mm256_loadu_si256((__m256i*)(src + i * width + j + 1));
			a[2] = _mm256_loadu_si256((__m256i*)(src + i * width + j - 1));

			a[3] = _mm256_loadu_si256((__m256i*)(src + (i + 1) * width + j));
			a[4] = _mm256_loadu_si256((__m256i*)(src + (i + 1) * width + j + 1));
			a[5] = _mm256_loadu_si256((__m256i*)(src + (i + 1) * width + j - 1));

			a[6] = _mm256_loadu_si256((__m256i*)(src + (i - 1) * width + j));
			a[7] = _mm256_loadu_si256((__m256i*)(src + (i - 1) * width + j + 1));
			a[8] = _mm256_loadu_si256((__m256i*)(src + (i - 1) * width + j - 1));

			for (int ii = 0; ii < 5; ii++) {
				for (int jj = ii + 1; jj < 9; jj++) {
					__m256i large = _mm256_max_epu8(a[ii], a[jj]);
					__m256i small = _mm256_min_epu8(a[ii], a[jj]);
					a[ii] = small;
					a[jj] = large;
				}
			}
			_mm256_storeu_si256((__m256i*)(dst + i * width + j), a[4]);
		}
		for (int je = j; je < width - 1; je++) {
			unsigned char a[9];
			a[0] = src[i * width + je];
			a[1] = src[i * width + je + 1];
			a[2] = src[i * width + je - 1];

			a[3] = src[(i + 1) * width + je];
			a[4] = src[(i + 1) * width + je + 1];
			a[5] = src[(i + 1) * width + je - 1];

			a[6] = src[(i - 1) * width + je];
			a[7] = src[(i - 1) * width + je + 1];
			a[8] = src[(i - 1) * width + je - 1];
			for (int ii = 0; ii < 5; ii++) {
				for (int jj = ii + 1; jj < 9; jj++) {
					unsigned char large = std::max<unsigned char>(a[ii], a[jj]);
					unsigned char small = std::min<unsigned char>(a[ii], a[jj]);
					a[ii] = small;
					a[jj] = large;
				}
			}
			dst[i * width + je] = a[4];
		}
	}
	for (int i = 0; i < width; i++) {
		dst[i] = src[i];
		dst[(height - 1) * width + i] = src[(height - 1) * width + i];
	}
	for (int i = 0; i < height; i++) {
		dst[i * width] = src[i * width];
		dst[i * width + width - 1] = src[i * width + width - 1];
	}
}

Mat AVX_MedianFilter(Mat src) {
	int row = src.rows;
	int col = src.cols;
	unsigned char * data = (unsigned char *)src.data;
	unsigned char *dst = new unsigned char[row * col];
	//for (int i = 0; i < row * col; i++) {
	//	printf("%d\n", data[i]);
	//}
	medianFilterAVX(row, col, data, dst);
	Mat res(row, col, CV_8UC1, dst);
	return res;
}

//使用Openmp加速中值滤波，在6核Core i7 3930K上，12线程的加速比为8.4

void MultiThtreadsFilter(int height, int width, unsigned char * __restrict src, unsigned char * __restrict dst) {
#pragma omp parallel default(none) shared(src, dst, row, col) num_threads(12)
{
#pragma omp for nowait 
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			unsigned char a[9];
			a[0] = src[i * width + j];
			a[1] = src[i * width + j + 1];
			a[2] = src[i * width + j - 1];

			a[3] = src[(i + 1) * width + j];
			a[4] = src[(i + 1) * width + j + 1];
			a[5] = src[(i + 1) * width + j - 1];

			a[6] = src[(i - 1) * width + j];
			a[7] = src[(i - 1) * width + j + 1];
			a[8] = src[(i - 1) * width + j - 1];
			for (int ii = 0; ii < 5; ii++) {
				for (int jj = ii + 1; jj < 9; jj++) {
					if (a[ii] > a[jj]) {
						unsigned char temp = a[ii];
						a[ii] = a[jj];
						a[jj] = temp;
					}
				}
			}
			dst[i * width + j] = a[4];
		}
	}
#pragma omp for nowait 
	for (int i = 0; i < width; i++) {
		dst[i] = src[i];
		dst[(height - 1) * width + i] = src[(height - 1) * width + i];
	}
#pragma omp for nowait 
	for (int i = 0; i < height; i++) {
		dst[i * width] = src[i * width];
		dst[i * width + width - 1] = src[i * width + width - 1];
	}
}
}

Mat Openmp_MedianFilter(Mat src) {
	int row = src.rows;
	int col = src.cols;
	unsigned char * data = (unsigned char *)src.data;
	unsigned char *dst = new unsigned char[row * col];
	//for (int i = 0; i < row * col; i++) {
	//	printf("%d\n", data[i]);
	//}
	MultiThtreadsFilter(row, col, data, dst);
	Mat res(row, col, CV_8UC1, dst);
	return res;
}

Mat speed_rgb2gray(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC1);
#pragma omp parallel for num_threads(4)
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<uchar>(i, j) = ((src.at<Vec3b>(i, j)[0] << 18) + (src.at<Vec3b>(i, j)[0] << 15) + (src.at<Vec3b>(i, j)[0] << 14) +
				(src.at<Vec3b>(i, j)[0] << 11) + (src.at<Vec3b>(i, j)[0] << 7) + (src.at<Vec3b>(i, j)[0] << 7) + (src.at<Vec3b>(i, j)[0] << 5) +
				(src.at<Vec3b>(i, j)[0] << 4) + (src.at<Vec3b>(i, j)[0] << 2) +
				(src.at<Vec3b>(i, j)[1] << 19) + (src.at<Vec3b>(i, j)[1] << 16) + (src.at<Vec3b>(i, j)[1] << 14) + (src.at<Vec3b>(i, j)[1] << 13) +
				(src.at<Vec3b>(i, j)[1] << 10) + (src.at<Vec3b>(i, j)[1] << 8) + (src.at<Vec3b>(i, j)[1] << 4) + (src.at<Vec3b>(i, j)[1] << 3) + (src.at<Vec3b>(i, j)[1] << 1) +
				(src.at<Vec3b>(i, j)[2] << 16) + (src.at<Vec3b>(i, j)[2] << 15) + (src.at<Vec3b>(i, j)[2] << 14) + (src.at<Vec3b>(i, j)[2] << 12) +
				(src.at<Vec3b>(i, j)[2] << 9) + (src.at<Vec3b>(i, j)[2] << 7) + (src.at<Vec3b>(i, j)[2] << 6) + (src.at<Vec3b>(i, j)[2] << 5) + (src.at<Vec3b>(i, j)[2] << 4) + (src.at<Vec3b>(i, j)[2] << 1) >> 20);
		}
	}
	return dst;
}

int main() {
	Mat src = cv::imread("F:\\2.png");
	src = speed_rgb2gray(src);
	int row = src.rows;
	int col = src.cols;
	Mat dst1 = speed_MedianFilter(src);
	cv::imshow("res1", dst1);
	Mat dst2 = AVX_MedianFilter(src);
	cv::imshow("res2", dst2);
	Mat dst3 = Openmp_MedianFilter(src);
	cv::imshow("res3", dst3);
	cv::waitKey(0);
	return 0;
}