#include "CTSE.h"


static void help(){
	printf("\n Contextual Tracker with Structural Encoding\n "
		"tracks one object by analyzing frames from default camera\n"
		"or from a input video file or from a directory of images.\n"
		"Usage : \n"
		"./CTSE [--camera]=<use camera>, [--file]=<path to file>, [--directory]=<image files in directory>, [--center]=<BB Center Coordinates>, [--width]=<BB width>, [--height]=<BB height> \n");

}

const char* keys = {
	"{c  |camera      |true			   | use camera or not}"
	"{f  |file        |tree.avi        | movie file path  }"
	"{d  |directory   |Image directory | directory file path }"
	"{bcX |coordinatesX |BB Center X	   | BB's center coordinatesX}"
	"{bcY |coordinatesY |BB Center Y	   | BB's center coordinatesY}"
	"{w  |width		  |BB width        | BB width}"
	"{h  |height      |BB height       | BB height}"
};

int main(int argc, const char** argv){
	help();
	Ptr<xfeatures2d::SIFT> oDetector = xfeatures2d::SiftFeatureDetector::create(CTSE_DEFAULT_RETAIN_SIFT_FEATURES, CTSE_DEFAULT_OCTAVE_LAYERS, CTSE_DEFAULT_CONTRAST_THRESHOLD, CTSE_DEFAULT_EDGE_THRESHOLD, CTSE_DEFAULT_SIGMA);
	cv::BFMatcher oMatcher(NORM_L2,true);
	cv::VideoCapture oVideoInput;
	cv::VideoWriter oVideoOut;
	cv::Mat oInputFrame;
	std::vector<std::string> accumulator;
	std::vector<std::string>::iterator iter,iter1;	
	std::vector<KeyPoint> oDetectedKeys;
	cv::Point2f oBBCenter;
	cv::Point2f oBBTopLeftCoordinate, oBBBotRightCoordinate;  
	size_t nFrames=0, i=0 ; 
	size_t  n_BBWidth, n_BBHeight;

	cv::CommandLineParser parser(argc, argv, keys);
	const bool bCamera = parser.get<bool>("c");
	const string sVideoFilePath = parser.get<string>("f");
	const string sImageDirectoryPath = parser.get<string>("d");
	float fCenterX = parser.get<float>("bcX");
	float fCenterY = parser.get<float>("bcY");
	oBBCenter = Point2f(fCenterX,fCenterY) ;
	n_BBWidth = parser.get<size_t>("w");
	n_BBHeight = parser.get<size_t>("h");

	if(bCamera) {
		oVideoInput.open(0);
		oVideoInput.set(CV_CAP_PROP_POS_FRAMES,0); 
		oVideoInput >> oInputFrame;
		nFrames = oVideoInput.get(CV_CAP_PROP_POS_FRAMES);
	}

	else{

		oVideoInput.open(sImageDirectoryPath);

	}


	/* If video file is used

	else{

	oVideoInput.open(sVideoFilePath);

	}*/


	if( !oVideoInput.isOpened()) {

		if(bCamera)
			printf("Could not open default camera. \n");

		else 
			printf("No image directory exists at '%s'. \n", sImageDirectoryPath.c_str());
		return -1;
	}

	CTSE oCTSAlg;
	for(;;){
		oVideoInput >> oInputFrame;
		nFrames = oVideoInput.get(CV_CAP_PROP_POS_FRAMES);
		if(oInputFrame.empty())
		{	break;}
		else
		{
			oCTSAlg.process(oInputFrame,nFrames, oBBCenter, n_BBWidth, n_BBHeight, oDetector, oMatcher);

		}

	}
	return 0;
}

