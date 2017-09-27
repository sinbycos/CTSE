#pragma once
#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <stdio.h>
#include <fstream>
#include <string>

#include "KeyPointCTSE.h"
#include "UtilsTrack.h"

//! defines the default value for CTSE::m_nFeatures passed to CTSE::m_oSIFTdetector.detect()
#define CTSE_DEFAULT_RETAIN_SIFT_FEATURES (2000)
//! defines the default value for CTSE::int m_nOctaveLayers passed to CTSE::m_oSIFTdetector.detect()
#define CTSE_DEFAULT_OCTAVE_LAYERS (3)
//! defines the default value for CTSE::m_fContrastThreshold passed to CTSE::m_oSIFTdetector.detect()
#define CTSE_DEFAULT_CONTRAST_THRESHOLD (0.04)
//! defines the default value for CTSE::m_edgeThreshold passed to CTSE::m_oSIFTdetector.detect()
#define CTSE_DEFAULT_EDGE_THRESHOLD (10)
//! defines the default value for CTSE::m_sigma passed to CTSE::m_oSIFTdetector.detect()
#define CTSE_DEFAULT_SIGMA (1.6)
//! defines the default value for CTSE::m_fLearningRate passed to CTSE::adaptModel()
#define CTSE_DEFAULT_LEARNING_RATE (0.1)
//! defines the default value for ContextualStructureTrackerCTSE::BFMatcher passed to BFMatcher::radiusMatch()
#define CTSE_DEFAULT_RADIUS1 (200)
//! defines the default value for ContextualStructureTrackerCTSE::BFMatcher passed to BFMatcher::radiusMatch()
#define CTSE_DEFAULT_RADIUS2 (250)
//! defines the default value for ContextualStructureTrackerCTSE::voting passed for calculating fGaussian
#define CTSE_DEFAULT_SIGMA_FGAUSSIAN (200.0)
//! defines the default value for ContextualStructureTrackerCTSE::adaptKeyPoints passed for calculating fGaussian
#define CTSE_DEFAULT_PROXIMITY_FACTOR_CONSTANT (0.05)


class CTSE {
public:

	//! constructor1
	CTSE();

	//! destructor
	~CTSE();

	void process(cv::Mat& oInitImg, size_t& nFrames, Point2f oBBCenter, size_t n_BBWidth, size_t n_BBHeight, Ptr<xfeatures2d::SIFT> oDetector, cv::BFMatcher& oMatcher);


	//! computes SIFT descriptors

	cv::Mat computeDescriptor(const cv::Mat& oInitImg, std::vector<targetKeysInfo>& voStructuredKeyPoints);

	//! output center position by the tracker
	void predictCenter(std::vector< std::vector<float> > accum, float tRadius, std::vector<targetKeysInfo>& m_voFilteredKeyPoints);


	//! adapts the weight of the keypoints present in the appearance model for a target
	void adaptKeyPoints(std::vector<targetKeysInfo>& voFilteredKeyPoints, std::vector<targetKeysInfo>& voKeyPoints, cv::Point2f& oPredictedCenter, size_t oFrames);


	//! Voting by keypoints using encoded structure
	void voting(std::vector<targetKeysInfo>& voFilteredKeyPoints, std::vector<targetKeysInfo>& voKeyPoints, cv::Point2f& oPredictedCenter, cv::Mat oImage, size_t oFrames);

	void copyDescriptorStructuredKeyPoints(KeyPointCTSE& oKPCTSE, cv::Mat& oDescriptor);

	void filterMatches(vector<vector<DMatch>>& oMatches);

	void correspondingMatches(vector<vector<DMatch>>& oMatchesR);

	void setIndicatorZero(std::vector<targetKeysInfo>& oKeyPoints);
	void setPowerOne(std::vector<targetKeysInfo>& oKeyPoints);

	void drawOutput(cv::Rect& oROI, cv::Mat& oImage, size_t& nFrames);

	cv::Mat m_oImage1, m_oImage2, m_oImage3, m_oDescriptor1, m_oDescriptor2, m_oImg_matches1;


	//! Matched keypoints between two consecutive frames
	vector<DMatch> m_voMatchDes;
	vector<vector<DMatch>> m_vvoMatchesR;

	//! Filtered matches after ratio test
	std::vector< DMatch > m_voMatches, m_voGoodMatchesR;

	//! Training mat for the Keypoint Matcher
	vector<Mat> m_voTraining;


	//! Appearance Model Keypoints
	//std::vector<targetKeysInfo> m_voModel;


	//! Weight associated with a keypoint
	float m_fWeight;


	//! Sift Extractor Object
	//Ptr<xfeatures2d::SIFT> m_oExtractor;
	cv::Ptr<Feature2D> m_oExtractor = xfeatures2d::SIFT::create(CTSE_DEFAULT_RETAIN_SIFT_FEATURES, CTSE_DEFAULT_OCTAVE_LAYERS, CTSE_DEFAULT_CONTRAST_THRESHOLD, CTSE_DEFAULT_EDGE_THRESHOLD, CTSE_DEFAULT_SIGMA); 
	//! KeyPointCTSE class member
	KeyPointCTSE m_oKeyPointModel, m_oKeyPointNonModel;

	//! Coordinates for top left BB x and y and center's and Predicted Center by tracker 
	cv::Point2f m_oPredictedCenter;

protected:
	//! SIFT detector initializer values
	int m_nFeatures;
	int m_nOctaveLayers;
	double m_ContrastThreshold;
	double m_EdgeThreshold;
	double m_sigma;

};