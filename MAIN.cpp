#include "CTSE.h"
#include <iostream>
#include "opencv2/xfeatures2d/nonfree.hpp"
using namespace cv;
int main(int argc, const char** argv)
{
	try
	{
		Ptr<xfeatures2d::SIFT> oDetector = xfeatures2d::SiftFeatureDetector::create(CTSE_DEFAULT_RETAIN_SIFT_FEATURES, CTSE_DEFAULT_OCTAVE_LAYERS, CTSE_DEFAULT_CONTRAST_THRESHOLD, CTSE_DEFAULT_EDGE_THRESHOLD, CTSE_DEFAULT_SIGMA);
		cv::BFMatcher oMatcher(NORM_L2, true);
		std::vector<std::string> accumulator;
		std::vector<std::string>::iterator iter, iter1;
		std::vector<KeyPoint> oDetectedKeys;
		cv::Point2f oBBCenter;
		cv::Point2f oBBTopLeftCoordinate, oBBBotRightCoordinate;
		size_t nFrames = 0, i = 0;
		size_t  n_BBWidth, n_BBHeight;
		int frame_counter = 0;
		cv::Rect roi(647, 30, 42, 42);
		Mat frame;
		//std::string video = "--Video Pathp4";
		std::string video = "2.mp4";
		VideoCapture cap(video);
		int times = 8;
		namedWindow("tracker", 0);
		resizeWindow("tracker", 100 * times, 50 * times);
		if (roi.width == 0 || roi.height == 0)
			return 0;
		CTSE oCTSAlg;
		oBBCenter.x = roi.x + roi.width / 2;
		oBBCenter.y = roi.y + roi.height / 2;
		printf("Start the tracking process\n");
		for (;; )
		{
			cap >> frame;
			nFrames = cap.get(CV_CAP_PROP_POS_FRAMES);
			if (frame.rows == 0 || frame.cols == 0)
				break;
			oCTSAlg.process(frame, nFrames, oBBCenter, roi.width, roi.height, oDetector, oMatcher);
		}
	}
	catch (const cv::Exception& et) {
		std::cerr << "top level caught cv::Exception:\n" << et.what() << std::endl;
	}
	catch (const std::exception& et) {
		std::cerr << "top level caught std::exception:\n" << et.what() << std::endl;
	}
	catch (...) {
		std::cerr << "top level caught unknown exception." << std::endl;
	}
	system("paused");
}



