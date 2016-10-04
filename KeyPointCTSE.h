#pragma once
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"


//! defines the default value for keypoint weight in KeyPointCTSE::createStructuralConfiguration
#define CTSE_DEFAULT_KEYPOINT_WEIGHT (0.2)

using namespace std;
using namespace cv;
#define e 2.71828

struct targetKeysInfo
{
	cv::KeyPoint oKey;
	cv::Point2f fDisFromCen, fPredCen, fMinusDis;
	float fDistance;
	float fWeight;
	float fPower;
	float fInitPower;
	float fLastPower;
	cv::Mat oDescriptor;
	size_t nIndex;
	size_t nIndi;
			
};


class KeyPointCTSE {

public:

//! full constructor
	KeyPointCTSE();

//! default destructor
	virtual ~KeyPointCTSE();

//! KeyPoints in the Model
void filteredKeyPoints(std::vector<targetKeysInfo>& voKeyPointsImg, cv::Rect& oROI);

//! Structural Configuration of the keypoints
void createStructuralConfiguration(std::vector<KeyPoint>& voKeyPointsImg);

void encodeStructure(std::vector<targetKeysInfo>& voFilteredKeyPoints, cv::Point2f& oCenter, cv::Mat image1);

void setROI(cv::Point2f& oROICenter, size_t nWidth, size_t nHeight);

void setROI2(cv::Point2f& oTopLeftBBCoordinate, size_t nWidth, size_t nHeight);

cv::Rect getROI();

cv::Point2f getROICenter();

cv::Point2f getTopLeftROICoordinate();

cv::Point2f getBotRightROICoordinate();


std::vector<targetKeysInfo> getStructuredKeyPoints();

std::vector<targetKeysInfo> m_voStructuredKeyPoints;

protected:

//! ROI initialized
	cv::Rect m_oROI;

//! Coordinates for top left BB x and y and center's and Predicted Center by tracker 
	cv::Point2f m_oROICenter, m_oPredCenter, m_oTopLeftROICoordinate, m_oBotRightROICoordinate;

};
