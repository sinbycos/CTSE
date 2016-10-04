#include "KeyPointCTSE.h"

KeyPointCTSE::KeyPointCTSE()
{
}


KeyPointCTSE::~KeyPointCTSE()
{
}

void KeyPointCTSE::setROI(cv::Point2f& oROICenter, size_t nWidth, size_t nHeight){

	m_oROICenter = oROICenter;
	m_oROI.width = nWidth;
	m_oROI.height = nHeight;
	m_oTopLeftROICoordinate = cv::Point2f(oROICenter.x - nWidth/2, oROICenter.y - nHeight/2);
	m_oBotRightROICoordinate = cv::Point2f((oROICenter.x + nWidth/2), ((1.2*oROICenter.y) + (nHeight/2)));
	m_oROI.x = m_oTopLeftROICoordinate.x;
	m_oROI.y = m_oTopLeftROICoordinate.y;
}


void KeyPointCTSE::setROI2(cv::Point2f& oTopLeftBBCoordinate, size_t nWidth, size_t nHeight){
	
	m_oROI.x = oTopLeftBBCoordinate.x;
	m_oROI.y = oTopLeftBBCoordinate.y;
	m_oROI.width = nWidth;
	m_oROI.height = nHeight;
	m_oROICenter.x = oTopLeftBBCoordinate.x  + nWidth/2;
	m_oROICenter.y = oTopLeftBBCoordinate.y + nHeight/2;
	m_oTopLeftROICoordinate = cv::Point2f(m_oROICenter.x - nWidth/2, m_oROICenter.y - nHeight/2);
	m_oBotRightROICoordinate = cv::Point2f((m_oROICenter.x + nWidth/2), ((1.2*m_oROICenter.y) + (nHeight/2)));
}


cv::Rect KeyPointCTSE::getROI(){

		return m_oROI;

}

cv::Point2f KeyPointCTSE::getROICenter(){

	return m_oROICenter;

}

cv::Point2f KeyPointCTSE::getTopLeftROICoordinate(){
	return m_oTopLeftROICoordinate;

}

cv::Point2f KeyPointCTSE::getBotRightROICoordinate(){
	
	return m_oBotRightROICoordinate;
}


void KeyPointCTSE::filteredKeyPoints(std::vector<targetKeysInfo>& voKeyPointsImg, cv::Rect& oROI){

	std::vector<targetKeysInfo> voFilteredKeyPoints;

	for(std::vector<targetKeysInfo>::iterator iter = voKeyPointsImg.begin(); iter!= voKeyPointsImg.end(); ++iter)
	{
		if (iter->oKey.pt.x >= oROI.x && iter->oKey.pt.x <= (oROI.x + oROI.width) && iter->oKey.pt.y >= oROI.y && iter->oKey.pt.y <= ((1.2*oROI.y) + (oROI.height))){

			voFilteredKeyPoints.push_back(*iter);


		}

	}

	voKeyPointsImg.clear();

	for(auto i = 0; i < voFilteredKeyPoints.size(); ++i)
	
	{
		voKeyPointsImg.push_back(voFilteredKeyPoints[i]);
	}


}



void KeyPointCTSE::createStructuralConfiguration(std::vector<KeyPoint>& voKeyPointsImg){

	m_voStructuredKeyPoints.resize(voKeyPointsImg.size());
	for(auto i=0; i < voKeyPointsImg.size(); i++){
	
		m_voStructuredKeyPoints[i].oKey = voKeyPointsImg[i];
		m_voStructuredKeyPoints[i].fDistance = 0;
		m_voStructuredKeyPoints[i].fWeight = CTSE_DEFAULT_KEYPOINT_WEIGHT; // weight past
		m_voStructuredKeyPoints[i].fDisFromCen = cv::Point2f(0.0,0.0);
		m_voStructuredKeyPoints[i].oDescriptor = cv::Mat(1,128,CV_32FC1);
		m_voStructuredKeyPoints[i].nIndi = -1;
		m_voStructuredKeyPoints[i].nIndex = -1;
		m_voStructuredKeyPoints[i].fPower = -1; // weight current
		
	}
}



std::vector<targetKeysInfo> KeyPointCTSE::getStructuredKeyPoints(){


	return m_voStructuredKeyPoints;

}




void KeyPointCTSE::encodeStructure(std::vector<targetKeysInfo>& voFilteredKeyPoints, cv::Point2f& oCenter, cv::Mat image1){
	

	for(unsigned int i = 0; i < voFilteredKeyPoints.size(); i++)
	{

		cv::Point2f oDisCenTemp =   oCenter - voFilteredKeyPoints[i].oKey.pt;
		voFilteredKeyPoints[i].fDisFromCen = oDisCenTemp;	

		
	}

}










