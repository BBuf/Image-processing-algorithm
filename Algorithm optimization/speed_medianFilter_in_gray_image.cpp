#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
//所有代码针对灰度图，RGB分为3个通道处理
//中值滤波串行代码
Mat speed_MedianFilter(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC1);
	for (int i = 1; i < row - 1; i++) {
		for (int j = 1; j < col - 1; j++) {
			unsigned char a[9];
			a[0] = src.at<uchar>(i, j);
			a[1] = src.at<uchar>(i, j + 1);
			a[2] = src.at<uchar>(i, j - 1);

			a[3] = src.at<uchar>(i + 1, j);
			a[4] = src.at<uchar>(i + 1, j + 1);
			a[5] = src.at<uchar>(i + 1, j - 1);

			a[6] = src.at<uchar>(i - 1, j);
			a[7] = src.at<uchar>(i - 1, j + 1);
			a[8] = src.at<uchar>(i - 1, j - 1);
			for (int ii = 0; ii < 5; ii++) {
				for (int jj = ii + 1; jj < 9; jj++) {
					if (a[ii] > a[jj]) {
						unsigned char tmp = a[ii];
						a[jj] = a[ii];
						a[jj] = tmp;
					}
				}
			}
			dst.at<uchar>(i, j) = a[4];
		}
	}
	for (int i = 0; i < row; i++) {
		dst.at<uchar>(i, 0) = src.at<uchar>(i, 0);
		dst.at<uchar>(i, col - 1) = src.at<uchar>(i, col - 1);
	}
	for (int i = 0; i < col; i++) {
		dst.at<uchar>(0, i) = src.at<uchar>(0, i);
		dst.at<uchar>(row - 1, i) = src.at<uchar>(row - 1, i);
	}
	return  dst;
}

//使用AVX指令集优化中值滤波

Mat AVX_MedianFilter(Mat src, unsigned char * __restrict temp) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	unsigned char * __restrict temp2 = new unsigned char[row * col];
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			temp2[i * col + j] = src.at<uchar>(i, j);
		}
	}
	for (int i = 1; i < row - 1; i++) {
		int j;
		for (j = 1; j < col - 1 - 32; j += 32) {
			__m256i a[9];
			a[0] = _mm256_loadu_si256((__m256i*)(temp2 + i * col + j));
			a[1] = _mm256_loadu_si256((__m256i*)(temp2 + i * col + j + 1));
			a[2] = _mm256_loadu_si256((__m256i*)(temp2 + i * col + j - 1));

			a[3] = _mm256_loadu_si256((__m256i*)(temp2 + (i + 1) * col + j));
			a[4] = _mm256_loadu_si256((__m256i*)(temp2 + (i + 1) * col + j + 1));
			a[5] = _mm256_loadu_si256((__m256i*)(temp2 + (i + 1) * col + j - 1));

			a[6] = _mm256_loadu_si256((__m256i*)(temp2 + (i - 1) * col + j));
			a[7] = _mm256_loadu_si256((__m256i*)(temp2 + (i - 1) * col + j + 1));
			a[8] = _mm256_loadu_si256((__m256i*)(temp2 + (i - 1) * col + j - 1));

			for (int ii = 0; ii < 5; ii++) {
				for (int jj = ii + 1; jj < 9; jj++) {
					__m256i larger = _mm256_max_epu8(a[ii], a[jj]);
					__m256i smaller = _mm256_min_epu8(a[ii], a[jj]);
					a[ii] = smaller;
					a[jj] = larger;
				}
			}
			_mm256_storeu_si256((__m256i*)(temp + i * col + j), a[4]);
		}
		for (int je = j; je < col - 1; je++) {
			unsigned char a[9];
			a[0] = src.at<uchar>(i, je);
			a[1] = src.at<uchar>(i, je + 1);
			a[2] = src.at<uchar>(i, je - 1);

			a[3] = src.at<uchar>(i + 1, je);
			a[4] = src.at<uchar>(i + 1, je + 1);
			a[5] = src.at<uchar>(i + 1, je - 1);

			a[6] = src.at<uchar>(i - 1, je);
			a[7] = src.at<uchar>(i - 1, je + 1);
			a[8] = src.at<uchar>(i - 1, je - 1);
			for (int ii = 0; ii < 5; ii++) {
				for (int jj = ii + 1; jj < 9; jj++) {
					unsigned char large = std::max<unsigned char>(a[ii], a[jj]);
					unsigned char small = std::min<unsigned char>(a[ii], a[jj]);
					a[ii] = small;
					a[jj] = large;
				}
			}
			temp[i * col + je] = a[4];
		}
	}
	for (int i = 1; i < row - 1; i++) {
		for (int j = 1; j < col - 1; j++) {
			dst.at<uchar>(i, j) = (temp[i * col + j]);
		}
	}
	for (int i = 0; i < row; i++) {
		dst.at<uchar>(i, 0) = src.at<uchar>(i, 0);
		dst.at<uchar>(i, col - 1) = src.at<uchar>(i, col - 1);
	}
	for (int i = 0; i < col; i++) {
		dst.at<uchar>(0, i) = src.at<uchar>(0, i);
		dst.at<uchar>(row - 1, i) = src.at<uchar>(row - 1, i);
	}
	return dst;
}

//使用Openmp加速中值滤波，在6核Core i7 3930K上，12线程的加速比为8.4

Mat Openmp_MedianFilter(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC1);
#pragma omp parallel default(none) shared(src, dst, row, col) num_threads(12)
	{
#pragma omp for nowait 
		for (int i = 1; i < row - 1; i++) {
			for (int j = 1; j < col - 1; j++) {
				unsigned char a[9];
				a[0] = src.at<uchar>(i, j);
				a[1] = src.at<uchar>(i, j + 1);
				a[2] = src.at<uchar>(i, j - 1);

				a[3] = src.at<uchar>(i + 1, j);
				a[4] = src.at<uchar>(i + 1, j + 1);
				a[5] = src.at<uchar>(i + 1, j - 1);

				a[6] = src.at<uchar>(i - 1, j);
				a[7] = src.at<uchar>(i - 1, j + 1);
				a[8] = src.at<uchar>(i - 1, j - 1);
				for (int ii = 0; ii < 5; ii++) {
					for (int jj = ii + 1; jj < 9; jj++) {
						if (a[ii] > a[jj]) {
							unsigned char tmp = a[ii];
							a[jj] = a[ii];
							a[jj] = tmp;
						}
					}
				}
				dst.at<uchar>(i, j) = a[4];
			}
		}
#pragma omp for nowait
		for (int i = 0; i < row; i++) {
			dst.at<uchar>(i, 0) = src.at<uchar>(i, 0);
			dst.at<uchar>(i, col - 1) = src.at<uchar>(i, col - 1);
		}
#pragma omp for nowait
		for (int i = 0; i < col; i++) {
			dst.at<uchar>(0, i) = src.at<uchar>(0, i);
			dst.at<uchar>(row - 1, i) = src.at<uchar>(row - 1, i);
		}
	}
	return  dst;
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
	Mat src = cv::imread("F:\\1.jpg");
	src = speed_rgb2gray(src);
	int row = src.rows;
	int col = src.cols;
	unsigned char * __restrict temp = new unsigned char[row * col];
	Mat dst1 = speed_MedianFilter(src);
	cv::imshow("res1", dst1);
	Mat dst2 = AVX_MedianFilter(src, temp);
	cv::imshow("res2", dst2);
	Mat dst3 = Openmp_MedianFilter(src);
	cv::imshow("res3", dst3);
	cv::waitKey(0);
	return 0;
}