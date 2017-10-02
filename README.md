
This directory contains a CTSE ( Paper titled 'Contextual Object Tracker with Structure Encoding' , ICIP 2015) object tracking algorithm in video sequences.

The main class used for tracking is CTSE.
The other class KeyPoint CTSE is used to keep the keypoint's information and the CTSE class interacts wth this class by having an KeyPointCTSE object as a member variable
The algorithm has dependency of OpenCV library and OpenCV-contrib. It can run on opencv versions from 3.0.0 to 3.2.0. 


Working Structure:

Input
=======
1. Directory of Images/ Video AVI file
2. Bounding Box centroid
3. Bounding Box Width
4. Bounding Box Height

Algorithm
=========
```
CTSE oCTSE

for(all frames in the video) {
    ...
oCTSE.process(input params);
    ...
}

```

Output
======
Blue Colored Bounding Box 

-----------
CTSE (Contextual Object Tracker with Structure Encoding)
Copyright (c) Tanushri Chakravorty (tanushri.chakravorty@polymtl.ca). All rights reserved.

See LICENSE.txt for more details.

