void dehazing::AirlightEstimation(IplImage* imInput)
{
    int nMinDistance = 65536;
    int nDistance;

    int nX, nY;

    int nMaxIndex;
    double dpScore[3];
    double dpMean[3];
    double dpStds[3];

    float afMean[4] = {0};
    float afScore[4] = {0};
    float nMaxScore = 0;

    int nWid = imInput->width;
    int nHei = imInput->height;

    int nStep = imInput->widthStep;

    // 4 sub-block
    IplImage *iplUpperLeft = cvCreateImage(cvSize(nWid/2, nHei/2),IPL_DEPTH_8U, 3);
    IplImage *iplUpperRight = cvCreateImage(cvSize(nWid/2, nHei/2),IPL_DEPTH_8U, 3);
    IplImage *iplLowerLeft = cvCreateImage(cvSize(nWid/2, nHei/2),IPL_DEPTH_8U, 3);
    IplImage *iplLowerRight = cvCreateImage(cvSize(nWid/2, nHei/2),IPL_DEPTH_8U, 3);

    IplImage *iplR = cvCreateImage(cvSize(nWid/2, nHei/2),IPL_DEPTH_8U, 1);
    IplImage *iplG = cvCreateImage(cvSize(nWid/2, nHei/2),IPL_DEPTH_8U, 1);
    IplImage *iplB = cvCreateImage(cvSize(nWid/2, nHei/2),IPL_DEPTH_8U, 1);

    // divide 
    cvSetImageROI(imInput, cvRect(0, 0, nWid/2, nHei/2));
    cvCopyImage(imInput, iplUpperLeft);
    cvSetImageROI(imInput, cvRect(nWid/2+nWid%2, 0, nWid, nHei/2));
    cvCopyImage(imInput, iplUpperRight);
    cvSetImageROI(imInput, cvRect(0, nHei/2+nHei%2, nWid/2, nHei));
    cvCopyImage(imInput, iplLowerLeft);
    cvSetImageROI(imInput, cvRect(nWid/2+nWid%2, nHei/2+nHei%2, nWid, nHei));
    cvCopyImage(imInput, iplLowerRight);

    // compare to threshold(200) --> bigger than threshold, divide the block
    if(nHei*nWid > 200)
    {
        // compute the mean and std-dev in the sub-block
        // upper left sub-blwww.shanxiwang.netock
        cvCvtPixToPlane(iplUpperLeft, iplR, iplG, iplB, 0);

        cvMean_StdDev(iplR, dpMean, dpStds);
        cvMean_StdDev(iplG, dpMean+1, dpStds+1);
        cvMean_StdDev(iplB, dpMean+2, dpStds+2);
        // dpScore: mean - std-dev
        dpScore[0] = dpMean[0]-dpStds[0];
        dpScore[1] = dpMean[1]-dpStds[1];
        dpScore[2] = dpMean[2]-dpStds[2];

        afScore[0] = (float)(dpScore[0]+dpScore[1]+dpScore[2]);

        nMaxScore = afScore[0];
        nMaxIndex = 0;

        // upper right sub-block
        cvCvtPixToPlane(iplUpperRight, iplR, iplG, iplB, 0);

        cvMean_StdDev(iplR, dpMean, dpStds);
        cvMean_StdDev(iplG, dpMean+1, dpStds+1);
        cvMean_StdDev(iplB, dpMean+2, dpStds+2);

        dpScore[0] = dpMean[0]-dpStds[0];
        dpScore[1] = dpMean[1]-dpStds[1];
        dpScore[2] = dpMean[2]-dpStds[2];

        afScore[1] = (float)(dpScore[0]+dpScore[1]+dpScore[2]);

        if(afScore[1] > nMaxScore)
        {
            nMaxScore = afScore[1];
            nMaxIndex = 1;
        }

        // lower left sub-block
        cvCvtPixToPlane(iplLowerLeft, iplR, iplG, iplB, 0);

        cvMean_StdDev(iplR, dpMean, dpStds);
        cvMean_StdDev(iplG, dpMean+1, dpStds+1);
        cvMean_StdDev(iplB, dpMean+2, dpStds+2);

        dpScore[0] = dpMean[0]-dpStds[0];
        dpScore[1] = dpMean[1]-dpStds[1];
        dpScore[2] = dpMean[2]-dpStds[2];

        afScore[2] = (float)(dpScore[0]+dpScore[1]+dpScore[2]);

        if(afScore[2] > nMaxScore)
        {
            nMaxScore = afScore[2];
            nMaxIndex = 2;
        }

        // lower right sub-block
        cvCvtPixToPlane(iplLowerRight, iplR, iplG, iplB, 0);

        cvMean_StdDev(iplR, dpMean, dpStds);
        cvMean_StdDev(iplG, dpMean+1, dpStds+1);
        cvMean_StdDev(iplB, dpMean+2, dpStds+2);

        dpScore[0] = dpMean[0]-dpStds[0];
        dpScore[1] = dpMean[1]-dpStds[1];
        dpScore[2] = dpMean[2]-dpStds[2];

        afScore[3] = (float)(dpScore[0]+dpScore[1]+dpScore[2]);

        if(afScore[3] > nMaxScore)
        {
            nMaxScore = afScore[3];
            nMaxIndex = 3;
        }

        // select the sub-block, which has maximum score
        switch (nMaxIndex)
        {
        case 0:
            AirlightEstimation(iplUpperLeft); break;
        case 1:
            AirlightEstimation(iplUpperRight); break;
        case 2:
            AirlightEstimation(iplLowerLeft); break;
        case 3:
            AirlightEstimation(iplLowerRight); break;
        }
    }
    else
    {
        // select the atmospheric light value in the sub-block
        for(nY=0; nY<nHei; nY++)
        {
            for(nX=0; nX<nWid; nX++)
            {
                // 255-r, 255-g, 255-b
                nDistance = int(sqrt(float(255-(uchar)imInput->imageData[nY*nStep+nX*3])*float(255-(uchar)imInput->imageData[nY*nStep+nX*3])
                    +float(255-(uchar)imInput->imageData[nY*nStep+nX*3+1])*float(255-(uchar)imInput->imageData[nY*nStep+nX*3+1])
                    +float(255-(uchar)imInput->imageData[nY*nStep+nX*3+2])*float(255-(uchar)imInput->imageData[nY*nStep+nX*3+2])));
                if(nMinDistance > nDistance)
                {
                    nMinDistance = nDistance;
                    m_anAirlight[0] = (uchar)imInput->imageData[nY*nStep+nX*3];
                    m_anAirlight[1] = (uchar)imInput->imageData[nY*nStep+nX*3+1];
                    m_anAirlight[2] = (uchar)imInput->imageData[nY*nStep+nX*3+2];
                }
            }
        }
    }
    cvReleaseImage(&iplUpperLeft);
    cvReleaseImage(&iplUpperRight);
    cvReleaseImage(&iplLowerLeft);
    cvReleaseImage(&iplLowerRight);

    cvReleaseImage(&iplR);
    cvReleaseImage(&iplG);
    cvReleaseImage(&iplB);
}

float dehazing::NFTrsEstimationColor(int *pnImageR, int *pnImageG, int *pnImageB, int nStartX, int nStartY, int nWid, int nHei)
{
    int nCounter;   
    int nX, nY;     
    int nEndX;
    int nEndY;

    int nOutR, nOutG, nOutB;        
    int nSquaredOut;                
    int nSumofOuts;                 
    int nSumofSquaredOuts;          
    float fTrans, fOptTrs;          
    int nTrans;                     
    int nSumofSLoss;                
    float fCost, fMinCost, fMean;   
    int nNumberofPixels, nLossCount;

    nEndX = __min(nStartX+m_nTBlockSize, nWid); 
    nEndY = __min(nStartY+m_nTBlockSize, nHei); 

    nNumberofPixels = (nEndY-nStartY)*(nEndX-nStartX) * 3;  

    fTrans = 0.3f;  
    nTrans = 427;

    for(nCounter=0; nCounter<7; nCounter++)
    {
        nSumofSLoss = 0;
        nLossCount = 0;
        nSumofSquaredOuts = 0;
        nSumofOuts = 0;

        for(nY=nStartY; nY<nEndY; nY++)
        {
            for(nX=nStartX; nX<nEndX; nX++)
            {

                nOutB = ((pnImageB[nY*nWid+nX] - m_anAirlight[0])*nTrans + 128*m_anAirlight[0])>>7; // (I-A)/t + A --> ((I-A)*k*128 + A*128)/128
                nOutG = ((pnImageG[nY*nWid+nX] - m_anAirlight[1])*nTrans + 128*m_anAirlight[1])>>7;
                nOutR = ((pnImageR[nY*nWid+nX] - m_anAirlight[2])*nTrans + 128*m_anAirlight[2])>>7;     

                if(nOutR>255)
                {
                    nSumofSLoss += (nOutR - 255)*(nOutR - 255);
                    nLossCount++;
                }
                else if(nOutR < 0)
                {
                    nSumofSLoss += nOutR * nOutR;
                    nLossCount++;
                }
                if(nOutG>255)
                {
                    nSumofSLoss += (nOutG - 255)*(nOutG - 255);
                    nLossCount++;
                }
                else if(nOutG < 0)
                {
                    nSumofSLoss += nOutG * nOutG;
                    nLossCount++;
                }
                if(nOutB>255)
                {
                    nSumofSLoss += (nOutB - 255)*(nOutB - 255);
                    nLossCount++;
                }
                else if(nOutB < 0)
                {
                    nSumofSLoss += nOutB * nOutB;
                    nLossCount++;
                }
                nSumofSquaredOuts += nOutB * nOutB + nOutR * nOutR + nOutG * nOutG;;
                nSumofOuts += nOutR + nOutG + nOutB;
            }
        }
        fMean = (float)(nSumofOuts)/(float)(nNumberofPixels);  
        fCost = m_fLambda1 * (float)nSumofSLoss/(float)(nNumberofPixels) 
            - ((float)nSumofSquaredOuts/(float)nNumberofPixels - fMean*fMean); 

        if(nCounter==0 || fMinCost > fCost)
        {
            fMinCost = fCost;
            fOptTrs = fTrans;
        }

        fTrans += 0.1f;
        nTrans = (int)(1.0f/fTrans*128.0f);
    }
    return fOptTrs; 
}