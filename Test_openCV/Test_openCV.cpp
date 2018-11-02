// Test_openCV.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

//#include <opencv2\opencv.hpp>
//using namespace std;
//using namespace cv;
//
//
//int main(int argc, char* argv[])
//{
//	const char* imagename = "1.jpg";
//	Mat image = imread(imagename);
//	imshow("image", image);
//	waitKey();
//	return 0;
//}

//===================  OPENCV 使用surf ===================//
// https://blog.csdn.net/lv1247736542/article/details/80312789
// https://blog.csdn.net/lv1247736542/article/details/80309984
//#include <iostream>
//#include <vector>
//#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/highgui.hpp>
//
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	Mat img1 = imread("pic1.jpg", 1);
//	Mat img2 = imread("pic2.jpg", 1);
//	if ((img1.data == NULL) || (img2.data == NULL))
//	{
//		cout << "No exist" << endl;
//		return -1;
//	}
//	Ptr<Feature2D> surf = xfeatures2d::SURF::create(1000);
//
//	vector<KeyPoint> keypoints_1, keypoints_2;
//	Mat descriptors_1, descriptors_2;
//
//	surf->detectAndCompute(img1, Mat(), keypoints_1, descriptors_1);
//	surf->detectAndCompute(img2, Mat(), keypoints_2, descriptors_2);
//	drawKeypoints(img1, keypoints_1, img1);
//	drawKeypoints(img2, keypoints_2, img2);
//
//	namedWindow("img1", 0);
//	resizeWindow("img1", 500, 500);
//	imshow("img1", img1);
//
//	namedWindow("img2", 0);
//	resizeWindow("img2", 500, 500);
//	imshow("img2", img2);
//
//	FlannBasedMatcher matcher;
//	std::vector< DMatch > matches;
//	matcher.match(descriptors_1, descriptors_2, matches);
//	double max_dist = 0; double min_dist = 100;
//
//	for (int i = 0; i < descriptors_1.rows; i++)
//	{
//		double dist = matches[i].distance;
//		if (dist < min_dist) min_dist = dist;
//		if (dist > max_dist) max_dist = dist;
//	}
//	printf("-- Max dist : %f \n", max_dist);
//	printf("-- Min dist : %f \n", min_dist);
//
//	std::vector< DMatch > good_matches;
//	for (int i = 0; i < descriptors_1.rows; i++)
//	{
//		if (matches[i].distance <= max(2 * min_dist, 0.02))
//		{
//			good_matches.push_back(matches[i]);
//		}
//	}
//
//	Mat img_matches;
//	drawMatches(img1, keypoints_1, img2, keypoints_2,
//		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//
//	namedWindow("Good Matches", 0);
//	resizeWindow("Good Matches", 800, 800);
//	imshow("Good Matches", img_matches);
//
//	for (int i = 0; i < (int)good_matches.size(); i++)
//	{
//		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n",
//			i, good_matches[i].queryIdx, good_matches[i].trainIdx);
//	}
//
//	waitKey(0);
//	return 0;
//}


//=============== 官方示例 cuda ===================//
//#include "stdafx.h"
//#include <iostream>
//#include "opencv2/opencv.hpp"
//#include <opencv2/core/cuda.hpp>
//
//using namespace std;
//using namespace cv;
//using namespace cv::cuda;
//int main()
//{
//	try
//	{
//		cout << getCudaEnabledDeviceCount() << endl;;
//	}
//	catch (const cv::Exception& ex)
//	{
//		cout << "Error:" << ex.what() << endl;
//	}
//	system("PAUSE");
//	return 0;
//}

//=============== 官方示例2 cuda ===================//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

int main(int argc, char* argv[])
{
	try
	{
		cv::Mat src_host = cv::imread("m-1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		cv::cuda::GpuMat dst, src;
		src.upload(src_host);

		cv::cuda::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);

		cv::Mat result_host;
		dst.download(result_host);

		cv::imshow("Result", result_host);
		cv::waitKey();
	}
	catch (const cv::Exception& ex)
	{
		std::cout << "Error: " << ex.what() << std::endl;
	}
	return 0;
}

//=============== 官方示例3 cuda gpu-basics-similarity===================//
//#include <iostream>                   // Console I/O
//#include <sstream>                    // String to number conversion
//
//#include <opencv2/core.hpp>      // Basic OpenCV structures
//#include <opencv2/core/utility.hpp>
//#include <opencv2/imgproc.hpp>// Image processing methods for the CPU
//#include <opencv2/imgcodecs.hpp>// Read images
//
//// CUDA structures and methods
//#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudafilters.hpp>
//
//using namespace std;
//using namespace cv;
//
//double getPSNR(const Mat& I1, const Mat& I2);      // CPU versions
//Scalar getMSSIM(const Mat& I1, const Mat& I2);
//
//double getPSNR_CUDA(const Mat& I1, const Mat& I2);  // Basic CUDA versions
//Scalar getMSSIM_CUDA(const Mat& I1, const Mat& I2);
//
////! [psnr]
//struct BufferPSNR                                     // Optimized CUDA versions
//{   // Data allocations are very expensive on CUDA. Use a buffer to solve: allocate once reuse later.
//	cuda::GpuMat gI1, gI2, gs, t1, t2;
//
//	cuda::GpuMat buf;
//};
////! [psnr]
//double getPSNR_CUDA_optimized(const Mat& I1, const Mat& I2, BufferPSNR& b);
//
////! [ssim]
//struct BufferMSSIM                                     // Optimized CUDA versions
//{   // Data allocations are very expensive on CUDA. Use a buffer to solve: allocate once reuse later.
//	cuda::GpuMat gI1, gI2, gs, t1, t2;
//
//	cuda::GpuMat I1_2, I2_2, I1_I2;
//	vector<cuda::GpuMat> vI1, vI2;
//
//	cuda::GpuMat mu1, mu2;
//	cuda::GpuMat mu1_2, mu2_2, mu1_mu2;
//
//	cuda::GpuMat sigma1_2, sigma2_2, sigma12;
//	cuda::GpuMat t3;
//
//	cuda::GpuMat ssim_map;
//
//	cuda::GpuMat buf;
//};
////! [ssim]
//Scalar getMSSIM_CUDA_optimized(const Mat& i1, const Mat& i2, BufferMSSIM& b);
//
//static void help()
//{
//	cout
//		<< "\n--------------------------------------------------------------------------" << endl
//		<< "This program shows how to port your CPU code to CUDA or write that from scratch." << endl
//		<< "You can see the performance improvement for the similarity check methods (PSNR and SSIM)." << endl
//		<< "Usage:" << endl
//		<< "./gpu-basics-similarity referenceImage comparedImage numberOfTimesToRunTest(like 10)." << endl
//		<< "--------------------------------------------------------------------------" << endl
//		<< endl;
//}
//
//int main(int, char *argv[])
//{
//	help();
//	Mat I1 = imread(argv[1]);           // Read the two images
//	Mat I2 = imread(argv[2]);
//
//	if (!I1.data || !I2.data)           // Check for success
//	{
//		cout << "Couldn't read the image";
//		return 0;
//	}
//
//	BufferPSNR bufferPSNR;
//	BufferMSSIM bufferMSSIM;
//
//	int TIMES = 10;
//	stringstream sstr(argv[3]);
//	sstr >> TIMES;
//	double time, result = 0;
//
//	//------------------------------- PSNR CPU ----------------------------------------------------
//	time = (double)getTickCount();
//
//	for (int i = 0; i < TIMES; ++i)
//		result = getPSNR(I1, I2);
//
//	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
//	time /= TIMES;
//
//	cout << "Time of PSNR CPU (averaged for " << TIMES << " runs): " << time << " milliseconds."
//		<< " With result of: " << result << endl;
//
//	//------------------------------- PSNR CUDA ----------------------------------------------------
//	time = (double)getTickCount();
//
//	for (int i = 0; i < TIMES; ++i)
//		result = getPSNR_CUDA(I1, I2);
//
//	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
//	time /= TIMES;
//
//	cout << "Time of PSNR CUDA (averaged for " << TIMES << " runs): " << time << " milliseconds."
//		<< " With result of: " << result << endl;
//
//	//------------------------------- PSNR CUDA Optimized--------------------------------------------
//	time = (double)getTickCount();                                  // Initial call
//	result = getPSNR_CUDA_optimized(I1, I2, bufferPSNR);
//	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
//	cout << "Initial call CUDA optimized:              " << time << " milliseconds."
//		<< " With result of: " << result << endl;
//
//	time = (double)getTickCount();
//	for (int i = 0; i < TIMES; ++i)
//		result = getPSNR_CUDA_optimized(I1, I2, bufferPSNR);
//
//	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
//	time /= TIMES;
//
//	cout << "Time of PSNR CUDA OPTIMIZED ( / " << TIMES << " runs): " << time
//		<< " milliseconds." << " With result of: " << result << endl << endl;
//
//
//	//------------------------------- SSIM CPU -----------------------------------------------------
//	Scalar x;
//	time = (double)getTickCount();
//
//	for (int i = 0; i < TIMES; ++i)
//		x = getMSSIM(I1, I2);
//
//	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
//	time /= TIMES;
//
//	cout << "Time of MSSIM CPU (averaged for " << TIMES << " runs): " << time << " milliseconds."
//		<< " With result of B" << x.val[0] << " G" << x.val[1] << " R" << x.val[2] << endl;
//
//	//------------------------------- SSIM CUDA -----------------------------------------------------
//	time = (double)getTickCount();
//
//	for (int i = 0; i < TIMES; ++i)
//		x = getMSSIM_CUDA(I1, I2);
//
//	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
//	time /= TIMES;
//
//	cout << "Time of MSSIM CUDA (averaged for " << TIMES << " runs): " << time << " milliseconds."
//		<< " With result of B" << x.val[0] << " G" << x.val[1] << " R" << x.val[2] << endl;
//
//	//------------------------------- SSIM CUDA Optimized--------------------------------------------
//	time = (double)getTickCount();
//	x = getMSSIM_CUDA_optimized(I1, I2, bufferMSSIM);
//	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
//	cout << "Time of MSSIM CUDA Initial Call            " << time << " milliseconds."
//		<< " With result of B" << x.val[0] << " G" << x.val[1] << " R" << x.val[2] << endl;
//
//	time = (double)getTickCount();
//
//	for (int i = 0; i < TIMES; ++i)
//		x = getMSSIM_CUDA_optimized(I1, I2, bufferMSSIM);
//
//	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
//	time /= TIMES;
//
//	cout << "Time of MSSIM CUDA OPTIMIZED ( / " << TIMES << " runs): " << time << " milliseconds."
//		<< " With result of B" << x.val[0] << " G" << x.val[1] << " R" << x.val[2] << endl << endl;
//	return 0;
//}
//
////! [getpsnr]
//double getPSNR(const Mat& I1, const Mat& I2)
//{
//	Mat s1;
//	absdiff(I1, I2, s1);       // |I1 - I2|
//	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
//	s1 = s1.mul(s1);           // |I1 - I2|^2
//
//	Scalar s = sum(s1);         // sum elements per channel
//
//	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
//
//	if (sse <= 1e-10) // for small values return zero
//		return 0;
//	else
//	{
//		double  mse = sse / (double)(I1.channels() * I1.total());
//		double psnr = 10.0*log10((255 * 255) / mse);
//		return psnr;
//	}
//}
////! [getpsnr]
//
////! [getpsnropt]
//double getPSNR_CUDA_optimized(const Mat& I1, const Mat& I2, BufferPSNR& b)
//{
//	b.gI1.upload(I1);
//	b.gI2.upload(I2);
//
//	b.gI1.convertTo(b.t1, CV_32F);
//	b.gI2.convertTo(b.t2, CV_32F);
//
//	cuda::absdiff(b.t1.reshape(1), b.t2.reshape(1), b.gs);
//	cuda::multiply(b.gs, b.gs, b.gs);
//
//	double sse = cuda::sum(b.gs, b.buf)[0];
//
//	if (sse <= 1e-10) // for small values return zero
//		return 0;
//	else
//	{
//		double mse = sse / (double)(I1.channels() * I1.total());
//		double psnr = 10.0*log10((255 * 255) / mse);
//		return psnr;
//	}
//}
////! [getpsnropt]
//
////! [getpsnrcuda]
//double getPSNR_CUDA(const Mat& I1, const Mat& I2)
//{
//	cuda::GpuMat gI1, gI2, gs, t1, t2;
//
//	gI1.upload(I1);
//	gI2.upload(I2);
//
//	gI1.convertTo(t1, CV_32F);
//	gI2.convertTo(t2, CV_32F);
//
//	cuda::absdiff(t1.reshape(1), t2.reshape(1), gs);
//	cuda::multiply(gs, gs, gs);
//
//	Scalar s = cuda::sum(gs);
//	double sse = s.val[0] + s.val[1] + s.val[2];
//
//	if (sse <= 1e-10) // for small values return zero
//		return 0;
//	else
//	{
//		double  mse = sse / (double)(gI1.channels() * I1.total());
//		double psnr = 10.0*log10((255 * 255) / mse);
//		return psnr;
//	}
//}
////! [getpsnrcuda]
//
////! [getssim]
//Scalar getMSSIM(const Mat& i1, const Mat& i2)
//{
//	const double C1 = 6.5025, C2 = 58.5225;
//	/***************************** INITS **********************************/
//	int d = CV_32F;
//
//	Mat I1, I2;
//	i1.convertTo(I1, d);           // cannot calculate on one byte large values
//	i2.convertTo(I2, d);
//
//	Mat I2_2 = I2.mul(I2);        // I2^2
//	Mat I1_2 = I1.mul(I1);        // I1^2
//	Mat I1_I2 = I1.mul(I2);        // I1 * I2
//
//								   /*************************** END INITS **********************************/
//
//	Mat mu1, mu2;   // PRELIMINARY COMPUTING
//	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
//	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
//
//	Mat mu1_2 = mu1.mul(mu1);
//	Mat mu2_2 = mu2.mul(mu2);
//	Mat mu1_mu2 = mu1.mul(mu2);
//
//	Mat sigma1_2, sigma2_2, sigma12;
//
//	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
//	sigma1_2 -= mu1_2;
//
//	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
//	sigma2_2 -= mu2_2;
//
//	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
//	sigma12 -= mu1_mu2;
//
//	///////////////////////////////// FORMULA ////////////////////////////////
//	Mat t1, t2, t3;
//
//	t1 = 2 * mu1_mu2 + C1;
//	t2 = 2 * sigma12 + C2;
//	t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
//
//	t1 = mu1_2 + mu2_2 + C1;
//	t2 = sigma1_2 + sigma2_2 + C2;
//	t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
//
//	Mat ssim_map;
//	divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
//
//	Scalar mssim = mean(ssim_map); // mssim = average of ssim map
//	return mssim;
//}
////! [getssim]
//
////! [getssimcuda]
//Scalar getMSSIM_CUDA(const Mat& i1, const Mat& i2)
//{
//	const float C1 = 6.5025f, C2 = 58.5225f;
//	/***************************** INITS **********************************/
//	cuda::GpuMat gI1, gI2, gs1, tmp1, tmp2;
//
//	gI1.upload(i1);
//	gI2.upload(i2);
//
//	gI1.convertTo(tmp1, CV_MAKE_TYPE(CV_32F, gI1.channels()));
//	gI2.convertTo(tmp2, CV_MAKE_TYPE(CV_32F, gI2.channels()));
//
//	vector<cuda::GpuMat> vI1, vI2;
//	cuda::split(tmp1, vI1);
//	cuda::split(tmp2, vI2);
//	Scalar mssim;
//
//	Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(vI2[0].type(), -1, Size(11, 11), 1.5);
//
//	for (int i = 0; i < gI1.channels(); ++i)
//	{
//		cuda::GpuMat I2_2, I1_2, I1_I2;
//
//		cuda::multiply(vI2[i], vI2[i], I2_2);        // I2^2
//		cuda::multiply(vI1[i], vI1[i], I1_2);        // I1^2
//		cuda::multiply(vI1[i], vI2[i], I1_I2);       // I1 * I2
//
//													 /*************************** END INITS **********************************/
//		cuda::GpuMat mu1, mu2;   // PRELIMINARY COMPUTING
//		gauss->apply(vI1[i], mu1);
//		gauss->apply(vI2[i], mu2);
//
//		cuda::GpuMat mu1_2, mu2_2, mu1_mu2;
//		cuda::multiply(mu1, mu1, mu1_2);
//		cuda::multiply(mu2, mu2, mu2_2);
//		cuda::multiply(mu1, mu2, mu1_mu2);
//
//		cuda::GpuMat sigma1_2, sigma2_2, sigma12;
//
//		gauss->apply(I1_2, sigma1_2);
//		cuda::subtract(sigma1_2, mu1_2, sigma1_2); // sigma1_2 -= mu1_2;
//
//		gauss->apply(I2_2, sigma2_2);
//		cuda::subtract(sigma2_2, mu2_2, sigma2_2); // sigma2_2 -= mu2_2;
//
//		gauss->apply(I1_I2, sigma12);
//		cuda::subtract(sigma12, mu1_mu2, sigma12); // sigma12 -= mu1_mu2;
//
//												   ///////////////////////////////// FORMULA ////////////////////////////////
//		cuda::GpuMat t1, t2, t3;
//
//		mu1_mu2.convertTo(t1, -1, 2, C1); // t1 = 2 * mu1_mu2 + C1;
//		sigma12.convertTo(t2, -1, 2, C2); // t2 = 2 * sigma12 + C2;
//		cuda::multiply(t1, t2, t3);        // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
//
//		cuda::addWeighted(mu1_2, 1.0, mu2_2, 1.0, C1, t1);       // t1 = mu1_2 + mu2_2 + C1;
//		cuda::addWeighted(sigma1_2, 1.0, sigma2_2, 1.0, C2, t2); // t2 = sigma1_2 + sigma2_2 + C2;
//		cuda::multiply(t1, t2, t1);                              // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
//
//		cuda::GpuMat ssim_map;
//		cuda::divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
//
//		Scalar s = cuda::sum(ssim_map);
//		mssim.val[i] = s.val[0] / (ssim_map.rows * ssim_map.cols);
//
//	}
//	return mssim;
//}
////! [getssimcuda]
//
////! [getssimopt]
//Scalar getMSSIM_CUDA_optimized(const Mat& i1, const Mat& i2, BufferMSSIM& b)
//{
//	const float C1 = 6.5025f, C2 = 58.5225f;
//	/***************************** INITS **********************************/
//
//	b.gI1.upload(i1);
//	b.gI2.upload(i2);
//
//	cuda::Stream stream;
//
//	b.gI1.convertTo(b.t1, CV_32F, stream);
//	b.gI2.convertTo(b.t2, CV_32F, stream);
//
//	cuda::split(b.t1, b.vI1, stream);
//	cuda::split(b.t2, b.vI2, stream);
//	Scalar mssim;
//
//	Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(b.vI1[0].type(), -1, Size(11, 11), 1.5);
//
//	for (int i = 0; i < b.gI1.channels(); ++i)
//	{
//		cuda::multiply(b.vI2[i], b.vI2[i], b.I2_2, 1, -1, stream);        // I2^2
//		cuda::multiply(b.vI1[i], b.vI1[i], b.I1_2, 1, -1, stream);        // I1^2
//		cuda::multiply(b.vI1[i], b.vI2[i], b.I1_I2, 1, -1, stream);       // I1 * I2
//
//		gauss->apply(b.vI1[i], b.mu1, stream);
//		gauss->apply(b.vI2[i], b.mu2, stream);
//
//		cuda::multiply(b.mu1, b.mu1, b.mu1_2, 1, -1, stream);
//		cuda::multiply(b.mu2, b.mu2, b.mu2_2, 1, -1, stream);
//		cuda::multiply(b.mu1, b.mu2, b.mu1_mu2, 1, -1, stream);
//
//		gauss->apply(b.I1_2, b.sigma1_2, stream);
//		cuda::subtract(b.sigma1_2, b.mu1_2, b.sigma1_2, cuda::GpuMat(), -1, stream);
//		//b.sigma1_2 -= b.mu1_2;  - This would result in an extra data transfer operation
//
//		gauss->apply(b.I2_2, b.sigma2_2, stream);
//		cuda::subtract(b.sigma2_2, b.mu2_2, b.sigma2_2, cuda::GpuMat(), -1, stream);
//		//b.sigma2_2 -= b.mu2_2;
//
//		gauss->apply(b.I1_I2, b.sigma12, stream);
//		cuda::subtract(b.sigma12, b.mu1_mu2, b.sigma12, cuda::GpuMat(), -1, stream);
//		//b.sigma12 -= b.mu1_mu2;
//
//		//here too it would be an extra data transfer due to call of operator*(Scalar, Mat)
//		cuda::multiply(b.mu1_mu2, 2, b.t1, 1, -1, stream); //b.t1 = 2 * b.mu1_mu2 + C1;
//		cuda::add(b.t1, C1, b.t1, cuda::GpuMat(), -1, stream);
//		cuda::multiply(b.sigma12, 2, b.t2, 1, -1, stream); //b.t2 = 2 * b.sigma12 + C2;
//		cuda::add(b.t2, C2, b.t2, cuda::GpuMat(), -12, stream);
//
//		cuda::multiply(b.t1, b.t2, b.t3, 1, -1, stream);     // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
//
//		cuda::add(b.mu1_2, b.mu2_2, b.t1, cuda::GpuMat(), -1, stream);
//		cuda::add(b.t1, C1, b.t1, cuda::GpuMat(), -1, stream);
//
//		cuda::add(b.sigma1_2, b.sigma2_2, b.t2, cuda::GpuMat(), -1, stream);
//		cuda::add(b.t2, C2, b.t2, cuda::GpuMat(), -1, stream);
//
//
//		cuda::multiply(b.t1, b.t2, b.t1, 1, -1, stream);     // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
//		cuda::divide(b.t3, b.t1, b.ssim_map, 1, -1, stream);      // ssim_map =  t3./t1;
//
//		stream.waitForCompletion();
//
//		Scalar s = cuda::sum(b.ssim_map, b.buf);
//		mssim.val[i] = s.val[0] / (b.ssim_map.rows * b.ssim_map.cols);
//
//	}
//	return mssim;
//}
////! [getssimopt]

//===================Histograms_Matching=========================//
/**
* @file BackProject_Demo1.cpp
* @brief Sample code for backproject function usage
* @author OpenCV team
*/

//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
///// Global Variables
//Mat src; Mat hsv; Mat hue;
//int bins = 25;
//
///// Function Headers
//void Hist_and_Backproj(int, void*);
//
//
///**
//* @function main
//*/
//int main(int, char** argv)
//{
//	/// Read the image
//	src = imread(argv[1], IMREAD_COLOR);
//
//	if (src.empty())
//	{
//		cout << "Usage: ./calcBackProject_Demo1 <path_to_image>" << endl;
//		return -1;
//	}
//
//	/// Transform it to HSV
//	cvtColor(src, hsv, COLOR_BGR2HSV);
//
//	/// Use only the Hue value
//	hue.create(hsv.size(), hsv.depth());
//	int ch[] = { 0, 0 };
//	mixChannels(&hsv, 1, &hue, 1, ch, 1);
//
//	/// Create Trackbar to enter the number of bins
//	const char* window_image = "Source image";
//	namedWindow(window_image, WINDOW_AUTOSIZE);
//	createTrackbar("* Hue  bins: ", window_image, &bins, 180, Hist_and_Backproj);
//	Hist_and_Backproj(0, 0);
//
//	/// Show the image
//	imshow(window_image, src);
//
//	/// Wait until user exits the program
//	waitKey(0);
//	return 0;
//}
//
//
///**
//* @function Hist_and_Backproj
//* @brief Callback to Trackbar
//*/
//void Hist_and_Backproj(int, void*)
//{
//	MatND hist;
//	int histSize = MAX(bins, 2);
//	float hue_range[] = { 0, 180 };
//	const float* ranges = { hue_range };
//
//	/// Get the Histogram and normalize it
//	calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
//	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
//
//	/// Get Backprojection
//	MatND backproj;
//	calcBackProject(&hue, 1, 0, hist, backproj, &ranges, 1, true);
//
//	/// Draw the backproj
//	imshow("BackProj", backproj);
//
//	/// Draw the histogram
//	int w = 400; int h = 400;
//	int bin_w = cvRound((double)w / histSize);
//	Mat histImg = Mat::zeros(w, h, CV_8UC3);
//
//	for (int i = 0; i < bins; i++)
//	{
//		rectangle(histImg, Point(i*bin_w, h), Point((i + 1)*bin_w, h - cvRound(hist.at<float>(i)*h / 255.0)), Scalar(0, 0, 255), -1);
//	}
//	imshow("Histogram", histImg);
//}

//====================== calculate histgram demo =======================//
/**
* @function calcHist_Demo.cpp
* @brief Demo code to use the function calcHist
* @author
*/

//#include "opencv2/highgui.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/imgproc.hpp"
//#include <iostream>
//
//using namespace std;
//using namespace cv;
//
///**
//* @function main
//*/
//int main(int argc, char** argv)
//{
//	Mat src, dst;
//
//	/// Load image
//	String imageName("data/lena.jpg"); // by default
//
//	if (argc > 1)
//	{
//		imageName = argv[1];
//	}
//
//	src = imread(imageName, IMREAD_COLOR);
//
//	if (src.empty())
//	{
//		return -1;
//	}
//
//	/// Separate the image in 3 places ( B, G and R )
//	vector<Mat> bgr_planes;
//	split(src, bgr_planes);
//
//	/// Establish the number of bins
//	int histSize = 256;
//
//	/// Set the ranges ( for B,G,R) )
//	float range[] = { 0, 256 };
//	const float* histRange = { range };
//
//	bool uniform = true; bool accumulate = false;
//
//	Mat b_hist, g_hist, r_hist;
//
//	/// Compute the histograms:
//	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
//	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
//	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
//
//	// Draw the histograms for B, G and R
//	int hist_w = 512; int hist_h = 400;
//	int bin_w = cvRound((double)hist_w / histSize);
//
//	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
//
//	/// Normalize the result to [ 0, histImage.rows ]
//	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//
//	/// Draw for each channel
//	for (int i = 1; i < histSize; i++)
//	{
//		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
//			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
//			Scalar(255, 0, 0), 2, 8, 0);
//		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
//			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
//			Scalar(0, 255, 0), 2, 8, 0);
//		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
//			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
//			Scalar(0, 0, 255), 2, 8, 0);
//	}
//
//	/// Display
//	namedWindow("calcHist Demo", WINDOW_AUTOSIZE);
//	imshow("calcHist Demo", histImage);
//
//	waitKey(0);
//
//	return 0;
//
//}
