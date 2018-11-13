#include "function.h"

void simple_color_balance(float ** input_img, float ** out_img, int rows, int cols) {
	float max_value = 0;
	float min_value = 256;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			max_value = max(max_value,input_img[i][j]);
			min_value = min(min_value, input_img[i][j]);
		}
	}
	if (max_value <= min_value) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				out_img[i][j] = max_value;
			}
		}
	}
	else {
		float scale = 255.0 / (max_value - min_value);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				if (input_img[i][j] < min_value) {
					out_img[i][j] = 0;
				}
				else if (input_img[i][j] > max_value) {
					out_img[i][j] = 255;
				}
				else {
					out_img[i][j] = scale * (input_img[i][j] - min_value);
				}
			}
		}
	}
}

int HDR(cv::Mat input_img, cv::Mat out_img) {
	int rows = input_img.rows;
	int cols = input_img.cols;
	//DouImg
	float ***DouImg;
	DouImg = new float **[rows];
	for (int i = 0; i < rows; i++) {
		DouImg[i] = new float *[cols];
		for (int j = 0; j < cols; j++) {
			DouImg[i][j] = new float[3];
		}
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			DouImg[i][j][0] = (float)input_img.at<Vec3b>(i, j)[0];
			DouImg[i][j][1] = (float)input_img.at<Vec3b>(i, j)[1];
			DouImg[i][j][2] = (float)input_img.at<Vec3b>(i, j)[2];
		}
	}
	//Lw
	float **Lw;
	Lw = new float *[rows];
	for (int i = 0; i < rows; i++) {
		Lw[i] = new float [cols];
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Lw[i][j] = 0;
		}
	}
	//B
	float **B;
	B = new float *[rows];
	for (int i = 0; i < rows; i++) {
		B[i] = new float[cols];
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			B[i][j] = (float)input_img.at<Vec3b>(i, j)[0];
		}
	}
	//G
	float **G;
	G = new float *[rows];
	for (int i = 0; i < rows; i++) {
		G[i] = new float[cols];
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			G[i][j] = (float)input_img.at<Vec3b>(i, j)[1];
		}
	}
	//R
	float **R;
	R = new float *[rows];
	for (int i = 0; i < rows; i++) {
		R[i] = new float[cols];
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			R[i][j] = (float)input_img.at<Vec3b>(i, j)[2];
		}
	}
	//Lwmax
	float Lwmax = 0.0;
	//Lw
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Lw[i][j] = 0.299 * R[i][j] + 0.587 * G[i][j] + 0.114 * B[i][j];
			if (Lw[i][j] == 0) {
				Lw[i][j] = 1;
			}
			Lwmax = max(Lw[i][j], Lwmax);
		}
	}
	//Lw_sum
	float Lw_sum = 0;
	//log_Lw
	float **log_Lw;
	log_Lw = new float *[rows];
	for (int i = 0; i < rows; i++) {
		log_Lw[i] = new float[cols];
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			log_Lw[i][j] = log(0.001 + Lw[i][j]);
			Lw_sum += log_Lw[i][j];
		}
	}
	//Lwaver
	float Lwaver = exp(Lw_sum / (rows * cols));
	//Lg
	float **Lg;
	Lg = new float *[rows];
	for (int i = 0; i < rows; i++) {
		Lg[i] = new float[cols];
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Lg[i][j] = log(Lw[i][j] / Lwaver + 1) / log(Lwmax / Lwaver + 1);
		}
	}
	//gain
	float **gain;
	gain = new float *[rows];
	for (int i = 0; i < rows; i++) {
		gain[i] = new float[cols];
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			gain[i][j] = Lg[i][j] / Lw[i][j];
		}
	}
	//gain*B, gain*G, gain*R
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			B[i][j] *= gain[i][j];
			G[i][j] *= gain[i][j];
			R[i][j] *= gain[i][j];
		}
	}
	simple_color_balance(B, B, rows, cols);
	simple_color_balance(G, G, rows, cols);
	simple_color_balance(R, R, rows, cols);

	for (int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++){
			out_img.at<Vec3b>(i, j)[0] = uchar((int)B[i][j]);
			out_img.at<Vec3b>(i, j)[1] = uchar((int)G[i][j]);
			out_img.at<Vec3b>(i, j)[2] = uchar((int)R[i][j]);
		}
	}
	//Free
	//DouImg
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			delete DouImg[i][j];
		}
		delete DouImg[i];
	}
	delete DouImg;
	//Lw
	for (int i = 0; i < rows; i++) {
		delete[] Lw[i];
	}
	delete Lw;
	//log_Lw
	for (int i = 0; i < rows; i++) {
		delete[] log_Lw[i];
	}
	delete log_Lw;
	//B
	for (int i = 0; i < rows; i++) {
		delete[] B[i];
	}
	delete B;
	//G
	for (int i = 0; i < rows; i++) {
		delete[] G[i];
	}
	delete G;
	//R
	for (int i = 0; i < rows; i++) {
		delete[] R[i];
	}
	delete R;
	//Lg
	for (int i = 0; i < rows; i++) {
		delete[] Lg[i];
	}
	delete Lg;
	//gain
	for (int i = 0; i < rows; i++) {
		delete[] gain[i];
	}
	delete gain;
	return 0;
}