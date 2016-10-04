#pragma once	
#define e 2.71828

// gaussian 5x5 pattern based on 'fspecial('gaussian', hsize, sigma)' 
// hsize 5 and sigma 3.0
static const int n_sizeOfGaussianPatternRows = 5;
static const int n_sizeOfGaussianPatternCols = 5;
static const int n_sizeOfGaussianPatternTot = 25;
static float f_GaussianMatrixArray[n_sizeOfGaussianPatternTot] = 
{0.0318  ,  0.0375    ,0.0397   , 0.0375,    0.0318,
0.0375,    0.0443    ,0.0469    ,0.0443    ,0.0375,
0.0397 ,   0.0469   , 0.0495    ,0.0469    ,0.0397,
0.0375  ,  0.0443  ,  0.0469    ,0.0443   , 0.0375,
0.0318   , 0.0375 ,   0.0397    ,0.0375  ,  0.0318,
};


static inline cv::Mat getGaussianMatrix(){

	cv::Mat oMatrix = cv::Mat(n_sizeOfGaussianPatternRows, n_sizeOfGaussianPatternCols, CV_32FC1, f_GaussianMatrixArray);
	return oMatrix;


}


