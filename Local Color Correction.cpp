Mat LCC(const Mat &src){
    int rows = src.rows;
    int cols = src.cols;
    int **I;
    I = new int *[rows];
    for(int i = 0; i < rows; i++){
        I[i] = new int [cols];
    }
    int **inv_I;
    inv_I = new int *[rows];
    for(int i = 0; i < rows; i++){
        inv_I[i] = new int [cols];
    }
    Mat Mast(rows, cols, CV_8UC1);
    for(int i = 0; i < rows; i++){
        uchar *data = Mast.ptr<uchar>(i);
        for(int j = 0; j < cols; j++){
            I[i][j] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[1]) / 3.0;
            inv_I[i][j] = 255;
            *data = inv_I[i][j] - I[i][j];
            data++;
        }
    }
    GaussianBlur(Mast, Mast, Size(41, 41), BORDER_DEFAULT);
    Mat dst(rows, cols, CV_8UC3);
    for(int i = 0; i < rows; i++){
        uchar *data = Mast.ptr<uchar>(i);
        for(int j = 0; j < cols; j++){
            for(int k = 0; k < 3; k++){
                float Exp = pow(2, (128 - data[j]) / 128.0);
                int value = int(255 * pow(src.at<Vec3b>(i, j)[k] / 255.0, Exp));
                dst.at<Vec3b>(i, j)[k] = value;
            }
        }
    }
    return dst;
}