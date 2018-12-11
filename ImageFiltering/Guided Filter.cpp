Mat getimage(Mat &a)
{
	int hei  =a.rows;
	int wid = a.cols;
	Mat I(hei, wid, CV_64FC1);
	//convert image depth to CV_64F
	a.convertTo(I, CV_64FC1,1.0/255.0);
	//normalize the pixel to 0~1
	/*
	for( int i = 0; i< hei; i++){
		double *p = I.ptr<double>(i);
		for( int j = 0; j< wid; j++){
			p[j] = p[j]/255.0; 	
		}
	}
	*/
	return I;
}
 
Mat cumsum(Mat &imSrc, int rc)
{
	if(!imSrc.data)
	{
		cout << "no data input!\n" << endl;
	}
	int hei = imSrc.rows;
	int wid = imSrc.cols;
	Mat imCum = imSrc.clone();
	if( rc == 1)
	{
		for( int i =1;i < hei; i++)
		{
			for( int j = 0; j< wid; j++)
			{
				imCum.at<double>(i,j) += imCum.at<double>(i-1,j);
			}
		}
	}
 
	if( rc == 2)
	{
		for( int i =0;i < hei; i++)
		{
			for( int j = 1; j< wid; j++)
			{
				imCum.at<double>(i,j) += imCum.at<double>(i,j-1);
			}
		}
	}
	return imCum;
}
 
Mat boxfilter(Mat &imSrc, int r)
{
	int hei = imSrc.rows;
	int wid = imSrc.cols;
	Mat imDst = Mat::zeros( hei, wid, CV_64FC1);
	//imCum = cumsum(imSrc, 1);
	Mat imCum = cumsum(imSrc,1);
	//imDst(1:r+1, :) = imCum(1+r:2*r+1, :);
	for( int i = 0; i<r+1; i++)
	{
		for( int j=0; j<wid; j++ )
		{
			imDst.at<double>(i,j) = imCum.at<double>(i+r,j);
		}
	}
	//imDst(r+2:hei-r, :) = imCum(2*r+2:hei, :) - imCum(1:hei-2*r-1, :);
	for( int i =r+1; i<hei-r;i++)
	{
		for( int j = 0; j<wid;j++)
		{
			imDst.at<double>(i,j) = imCum.at<double>(i+r,j)-imCum.at<double>(i-r-1,j);
		}
	}
	//imDst(hei-r+1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei-2*r:hei-r-1, :);
	for( int i = hei-r; i< hei; i++)
	{
		for( int j = 0; j< wid; j++)
		{
			imDst.at<double>(i,j) = imCum.at<double>(hei-1,j)-imCum.at<double>(i-r-1,j);
		}
	}
	imCum = cumsum(imDst, 2);
	//imDst(:, 1:r+1) = imCum(:, 1+r:2*r+1);
	for( int i = 0; i<hei; i++)
	{
		for( int j=0; j<r+1; j++ )
		{
			imDst.at<double>(i,j) = imCum.at<double>(i,j+r);
		}
	}
	//imDst(:, r+2:wid-r) = imCum(:, 2*r+2:wid) - imCum(:, 1:wid-2*r-1);
	for( int i =0 ; i<hei;i++)
	{
		for( int j = r+1; j<wid-r ;j++ )
		{
			imDst.at<double>(i,j) = imCum.at<double>(i,j+r)-imCum.at<double>(i,j-r-1);
		}
	}
	//imDst(:, wid-r+1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid-2*r:wid-r-1);
	for( int i = 0; i< hei; i++)
	{
		for( int j = wid-r; j<wid; j++)
		{
			imDst.at<double>(i,j) = imCum.at<double>(i,wid-1)-imCum.at<double>(i,j-r-1);
		}
	}
	return imDst;
}
 
Mat guidedfilter( Mat &I, Mat &p, int r, double eps ) 
{
	int hei = I.rows;
	int wid = I.cols;
	//N = boxfilter(ones(hei, wid), r);
	Mat one = Mat::ones(hei, wid, CV_64FC1);
	Mat N = boxfilter(one, r);
 
	//mean_I = boxfilter(I, r) ./ N;
	Mat mean_I(hei, wid, CV_64FC1);
	divide(boxfilter(I, r), N, mean_I);
 
	//mean_p = boxfilter(p, r) ./ N;
	Mat mean_p(hei, wid, CV_64FC1);
	divide(boxfilter(p, r), N, mean_p);
 
	//mean_Ip = boxfilter(I.*p, r) ./ N;
	Mat mul_Ip(hei, wid, CV_64FC1);
	Mat mean_Ip(hei, wid, CV_64FC1);
	multiply(I,p,mul_Ip);
	divide(boxfilter(mul_Ip, r), N, mean_Ip);
 
	//cov_Ip = mean_Ip - mean_I .* mean_p
	//this is the covariance of (I, p) in each local patch.
	Mat mul_mean_Ip(hei, wid, CV_64FC1);
	Mat cov_Ip(hei, wid, CV_64FC1);
	multiply(mean_I, mean_p, mul_mean_Ip);
	subtract(mean_Ip, mul_mean_Ip, cov_Ip);
 
	//mean_II = boxfilter(I.*I, r) ./ N;
	Mat mul_II(hei, wid, CV_64FC1);
	Mat mean_II(hei, wid, CV_64FC1);
	multiply(I, I, mul_II);
	divide(boxfilter(mul_II, r), N, mean_II);
 
	//var_I = mean_II - mean_I .* mean_I;
	Mat mul_mean_II(hei, wid, CV_64FC1);
	Mat var_I(hei, wid, CV_64FC1);
	multiply(mean_I, mean_I, mul_mean_II);
	subtract(mean_II, mul_mean_II, var_I);
 
	//a = cov_Ip ./ (var_I + eps);
	Mat a(hei, wid, CV_64FC1);
	for( int i = 0; i< hei; i++){
		double *p = var_I.ptr<double>(i);
		for( int j = 0; j< wid; j++){
			p[j] = p[j] +eps; 	
		}
	}
	divide(cov_Ip, var_I, a);
 
	//b = mean_p - a .* mean_I;
	Mat a_mean_I(hei ,wid, CV_64FC1);
	Mat b(hei ,wid, CV_64FC1);
	multiply(a, mean_I, a_mean_I);
	subtract(mean_p, a_mean_I, b);
 
	//mean_a = boxfilter(a, r) ./ N;
	Mat mean_a(hei, wid, CV_64FC1);
	divide(boxfilter(a, r), N, mean_a);
	//mean_b = boxfilter(b, r) ./ N;
	Mat mean_b(hei, wid, CV_64FC1);
	divide(boxfilter(b, r), N, mean_b);
 
	//q = mean_a .* I + mean_b;
	Mat mean_a_I(hei, wid, CV_64FC1);
	Mat q(hei, wid, CV_64FC1);
	multiply(mean_a, I, mean_a_I);
	add(mean_a_I, mean_b, q);
 
	return q;
}
 
/*****************
http://research.microsoft.com/en-us/um/people/kahe/eccv10/
推酷上的一篇文章：
http://www.tuicool.com/articles/Mv2iiu
************************/
cv::Mat guidedFilter2(cv::Mat I, cv::Mat p, int r, double eps)
{
  /*
  % GUIDEDFILTER   O(1) time implementation of guided filter.
  %
  %   - guidance image: I (should be a gray-scale/single channel image)
  %   - filtering input image: p (should be a gray-scale/single channel image)
  %   - local window radius: r
  %   - regularization parameter: eps
  */
 
  cv::Mat _I;
  I.convertTo(_I, CV_64FC1);
  I = _I;
 
  cv::Mat _p;
  p.convertTo(_p, CV_64FC1);
  p = _p;
 
  //[hei, wid] = size(I);
  int hei = I.rows;
  int wid = I.cols;
 
  //N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.
  cv::Mat N;
  cv::boxFilter(cv::Mat::ones(hei, wid, I.type()), N, CV_64FC1, cv::Size(r, r));
 
  //mean_I = boxfilter(I, r) ./ N;
  cv::Mat mean_I;
  cv::boxFilter(I, mean_I, CV_64FC1, cv::Size(r, r));
  
  //mean_p = boxfilter(p, r) ./ N;
  cv::Mat mean_p;
  cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));
 
  //mean_Ip = boxfilter(I.*p, r) ./ N;
  cv::Mat mean_Ip;
  cv::boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));
 
  //cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
  cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
 
  //mean_II = boxfilter(I.*I, r) ./ N;
  cv::Mat mean_II;
  cv::boxFilter(I.mul(I), mean_II, CV_64FC1, cv::Size(r, r));
 
  //var_I = mean_II - mean_I .* mean_I;
  cv::Mat var_I = mean_II - mean_I.mul(mean_I);
 
  //a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;	
  cv::Mat a = cov_Ip/(var_I + eps);
 
  //b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
  cv::Mat b = mean_p - a.mul(mean_I);
 
  //mean_a = boxfilter(a, r) ./ N;
  cv::Mat mean_a;
  cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
  mean_a = mean_a/N;
 
  //mean_b = boxfilter(b, r) ./ N;
  cv::Mat mean_b;
  cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
  mean_b = mean_b/N;
 
  //q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
  cv::Mat q = mean_a.mul(I) + mean_b;
 
  return q;
}