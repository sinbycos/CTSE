#include "CTSE.h"

CTSE::CTSE(){
}

CTSE::~CTSE(){
}

void CTSE::process(cv::Mat& oInitImg, size_t& nFrames, Point2f oBBCenter, size_t n_BBWidth, size_t n_BBHeight, cv::SiftFeatureDetector& oDetector, cv::BFMatcher& oMatcher){

	std::vector<KeyPoint> oDetectedKeys;
	cv::Mat oDescriptor2; std::vector<Mat> oTraining;
	vector<vector<DMatch>> vvoMatchesR;
	cv::Mat oInputFrameOne;
	cv::Mat oInputFrameOneCopy = oInitImg.clone();
	cv::Mat img_matches1;
	cv::Rect oOutputROI;
	cv::Point2f oPt1, oPt2;
	vector<KeyPoint> oDrawKeysModel;

	oInputFrameOne = oInitImg.clone();
	if (nFrames == 1)
	{	



		oDetector.detect(oInputFrameOne, oDetectedKeys);
		m_oKeyPointModel.createStructuralConfiguration(oDetectedKeys);
		setPowerOne(m_oKeyPointModel.m_voStructuredKeyPoints);
		//m_oKeyPointModel.setROI(oBBCenter, n_BBWidth, n_BBHeight);
		m_oKeyPointModel.setROI2(oBBCenter, n_BBWidth, n_BBHeight);
		m_oKeyPointModel.filteredKeyPoints(m_oKeyPointModel.m_voStructuredKeyPoints, m_oKeyPointModel.getROI());


		for(auto i = 0; i < m_oKeyPointModel.m_voStructuredKeyPoints.size();i++)
		{
			oDrawKeysModel.push_back(m_oKeyPointModel.m_voStructuredKeyPoints[i].oKey);

		}

		m_oDescriptor1 = computeDescriptor(oInputFrameOne, m_oKeyPointModel.m_voStructuredKeyPoints);			
		copyDescriptorStructuredKeyPoints(m_oKeyPointModel, m_oDescriptor1);
		m_oKeyPointModel.encodeStructure(m_oKeyPointModel.m_voStructuredKeyPoints, m_oKeyPointModel.getROICenter(), oInputFrameOneCopy);
		oTraining.push_back(m_oDescriptor1);
		oMatcher.add(oTraining);
		oMatcher.train();
	}

	if(nFrames == 2)
	{
		m_oPredictedCenter = m_oKeyPointModel.getROICenter();
	}

	if(nFrames != 1)
	{

		cv::Mat oInputFrameTwo = oInitImg.clone();
		cv::Mat oInputFrameTwoCopy = oInitImg.clone();
		cv::Mat oInputFrameTwoCopy2 = oInitImg.clone();
		oDetector.detect(oInputFrameTwo, oDetectedKeys);
		m_oKeyPointNonModel.createStructuralConfiguration(oDetectedKeys);
		oDescriptor2 = computeDescriptor(oInputFrameTwo,m_oKeyPointNonModel.m_voStructuredKeyPoints);
		copyDescriptorStructuredKeyPoints(m_oKeyPointNonModel, oDescriptor2);
		oMatcher.radiusMatch(m_oDescriptor1, oDescriptor2, vvoMatchesR, CTSE_DEFAULT_RADIUS2);
		setIndicatorZero(m_oKeyPointModel.m_voStructuredKeyPoints);
		filterMatches(vvoMatchesR);
		correspondingMatches(vvoMatchesR);
		voting(m_oKeyPointModel.m_voStructuredKeyPoints, m_oKeyPointNonModel.m_voStructuredKeyPoints, m_oPredictedCenter, oInputFrameTwoCopy, nFrames);
		oOutputROI.x = m_oPredictedCenter.x - (m_oKeyPointModel.getROI().width/2);
		oOutputROI.y = m_oPredictedCenter.y - (m_oKeyPointModel.getROI().height/2);
		oOutputROI.width = m_oKeyPointModel.getROI().width;
		oOutputROI.height = m_oKeyPointModel.getROI().height;
		m_oKeyPointModel.setROI(m_oPredictedCenter, oOutputROI.width, oOutputROI.height);
		drawOutput(oOutputROI, oInputFrameTwoCopy2, nFrames);
		adaptKeyPoints(m_oKeyPointModel.m_voStructuredKeyPoints, m_oKeyPointNonModel.m_voStructuredKeyPoints, m_oPredictedCenter, nFrames);
	

	}
}




cv::Mat CTSE::computeDescriptor(const cv::Mat& oInitImg, std::vector<targetKeysInfo>& voStructuredKeyPoints) {
	std::vector<KeyPoint> voSiftKeys;
	cv::Mat oDescriptor;

	for(auto i = 0; i < voStructuredKeyPoints.size(); ++i){

		voSiftKeys.push_back(voStructuredKeyPoints[i].oKey);
	}

	m_oExtractor.compute(oInitImg, voSiftKeys, oDescriptor);

	return oDescriptor;

}


void CTSE::copyDescriptorStructuredKeyPoints(KeyPointCTSE& oKPCTSE, Mat& oDescriptor){
	for (auto i= 0; i < oDescriptor.rows; i++){
		oKPCTSE.m_voStructuredKeyPoints[i].oDescriptor = oDescriptor.row(i).clone();
	}
}


void CTSE::filterMatches(vector<vector<DMatch>>& oMatches){

	for (size_t k = 0; k < oMatches.size(); ++k)
	{

		if(oMatches[k].size() > 1)
		{

			float ratio = oMatches[k][0].distance/ oMatches[k][1].distance;

			if(ratio > 0.8)
			{

				oMatches[k].clear();
			}
			
			else

					{
						cv::DMatch match = oMatches[k][0] ;

						oMatches[k].clear();

						oMatches[k].push_back(match);
					
					}


		}

	}					
}

void CTSE::correspondingMatches(vector<vector<DMatch>>& oMatchesR)
{

	for (size_t k = 0; k < oMatchesR.size(); k++)
	{

		if(!oMatchesR[k].empty()){
			cv::DMatch match = oMatchesR[k][0];
			m_oKeyPointModel.m_voStructuredKeyPoints.at(match.queryIdx).fDistance = match.distance;
			m_oKeyPointModel.m_voStructuredKeyPoints.at(match.queryIdx).nIndex = match.trainIdx;
			m_oKeyPointModel.m_voStructuredKeyPoints.at(match.queryIdx).nIndi = 1;
			m_oKeyPointNonModel.m_voStructuredKeyPoints[match.trainIdx].fDistance = match.distance;
			m_oKeyPointNonModel.m_voStructuredKeyPoints[match.trainIdx].nIndex = match.queryIdx;
			m_oKeyPointNonModel.m_voStructuredKeyPoints[match.trainIdx].nIndi = 1;
		}
	}
}


void CTSE::setIndicatorZero(std::vector<targetKeysInfo>& oKeyPoints){

	for(auto i=0; i < oKeyPoints.size(); i++)
	{

		oKeyPoints[i].nIndi = 0;
	}


}



void CTSE::setPowerOne(std::vector<targetKeysInfo>& oKeyPoints){

	for(auto i=0; i < oKeyPoints.size(); i++)
	{

		oKeyPoints[i].fPower = 1;
		
	}


}


void CTSE::voting(std::vector<targetKeysInfo>& voFilteredKeyPoints, std::vector<targetKeysInfo>& voKeyPoints, cv::Point2f& oPredictedCenter, cv::Mat oImage, size_t oFrames){


	cv::Mat test = oImage.clone();
	cv::Mat oGauss; size_t oLocationColoVotingMatrix, oLocationRowoVotingMatrix; 
	size_t oVotingMatrixBoundary = 20;
	int oRow=0, oCol =0;; //Voting does not happend outside the voting matrix

	cv::Point2f oDifference, oPixel ;

	float fDifferenceSquare, fGaussian, fGaussianFunction, fMinValVotingMatrix = 0.0; 

	oGauss = getGaussianMatrix();
	cv::Mat oVotingMatrix = Mat::zeros(oImage.rows, oImage.cols, CV_32FC1);



	for(unsigned int i = 0; i < voFilteredKeyPoints.size(); i++)
	{
		if(voFilteredKeyPoints[i].nIndi == 1)
		{
			
			voFilteredKeyPoints[i].fPredCen = voKeyPoints[voFilteredKeyPoints[i].nIndex].oKey.pt + voFilteredKeyPoints[i].fDisFromCen;

			oDifference =  voFilteredKeyPoints[i].fPredCen - oPredictedCenter;

			fDifferenceSquare = oDifference.x*oDifference.x + oDifference.y*oDifference.y;

			

			fGaussian = fDifferenceSquare/(2*pow(CTSE_DEFAULT_SIGMA_FGAUSSIAN,2));

			fGaussianFunction =  exp(-fGaussian);

			oLocationColoVotingMatrix = int(voFilteredKeyPoints[i].fPredCen.x);
			oLocationRowoVotingMatrix = int(voFilteredKeyPoints[i].fPredCen.y);

			if (oLocationColoVotingMatrix >= oVotingMatrixBoundary  && oLocationColoVotingMatrix <= (oVotingMatrix.cols - oVotingMatrixBoundary) && oLocationRowoVotingMatrix >= oVotingMatrixBoundary && oLocationRowoVotingMatrix <= (oVotingMatrix.rows - oVotingMatrixBoundary) )

			{
				cv::Mat oSubVotingMatrix = oVotingMatrix(cv::Rect(oLocationColoVotingMatrix- oGauss.rows/2, oLocationRowoVotingMatrix - oGauss.cols/2, oGauss.rows, oGauss.cols));
				oSubVotingMatrix +=  voFilteredKeyPoints[i].fWeight*oGauss*fGaussianFunction;

			}

		}

		

	}

	for(int j = 0 ; j < oVotingMatrix.rows; ++j)
	{

		for(int i = 0 ; i < oVotingMatrix.cols; ++i)
		{

			if(oVotingMatrix.at<float>(j,i) > fMinValVotingMatrix && oVotingMatrix.at<float>(j,i) > fMinValVotingMatrix ){

				fMinValVotingMatrix = oVotingMatrix.at<float>(j,i);
				oPixel = Point2f(j,i);
			}


		}

	}


	if(fMinValVotingMatrix > 0.0 )
	{
		int r = int(oPixel.x);
		int c = int(oPixel.y);

		oRow = r;
		oCol = c;
		oPredictedCenter = Point2f(oCol, oRow);

	}

	else
	{

		oPredictedCenter = oPredictedCenter;

	}

}


void CTSE::drawOutput(cv::Rect& oROI, cv::Mat& oImage, size_t& nFrames){
	cv::Point2f oPt1, oPt2;
	oPt1 = Point2f(oROI.x,oROI.y);
	oPt2 = Point2f((oROI.x + oROI.width), (oROI.y + (oROI.height)));

	cv::rectangle(oImage, oPt1, oPt2, cv::Scalar(255,0,0), 3, 8, 0);
	char frameString[10];
	char sym[2] ="#";
	itoa(nFrames, frameString, 10);
	strcat(frameString,sym);
	cv::putText(oImage, frameString, cv::Point(10,20), FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255,0,0));	
	namedWindow("CTSE", WINDOW_AUTOSIZE ); 
	imshow("CTSE", oImage);
	waitKey(10);

}

void CTSE::adaptKeyPoints(std::vector<targetKeysInfo>& voFilteredKeyPoints,  std::vector<targetKeysInfo>& voKeyPoints, cv::Point2f& oPredictedCenter, size_t oFrames){

	Point2f oErrorCenters; 
	float fSquareError, fProxFactor , fLearningRate = CTSE_DEFAULT_LEARNING_RATE; 

	for(unsigned int i = 0; i < voFilteredKeyPoints.size(); i++)
	{
		if ((voFilteredKeyPoints[i].nIndi == 1)){

			oErrorCenters = voFilteredKeyPoints[i].fPredCen - m_oPredictedCenter;
			fSquareError = sqrt(oErrorCenters.x *oErrorCenters.x + oErrorCenters.y*oErrorCenters.y); 
			fProxFactor = std::max<float>((1 - abs(CTSE_DEFAULT_PROXIMITY_FACTOR_CONSTANT*fSquareError)),0.0);
			
			voFilteredKeyPoints[i].fPower = fProxFactor ;
		

			if( fProxFactor == 0)
			{
				voFilteredKeyPoints[i].fWeight =  voFilteredKeyPoints[i].fWeight - (fLearningRate)*voFilteredKeyPoints[i].fWeight;

			}

			else

			{
				
				voFilteredKeyPoints[i].fWeight = (1 - fLearningRate)*voFilteredKeyPoints[i].fWeight + (fLearningRate)*voFilteredKeyPoints[i].fPower;
						
			}

			
		}

		else{ 

			voFilteredKeyPoints[i].fWeight = voFilteredKeyPoints[i].fWeight - (fLearningRate)*voFilteredKeyPoints[i].fWeight;


		}
	}
}

