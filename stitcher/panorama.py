# import the necessary packages
import numpy as np
import imutils
import cv2, time
class Stitcher:
    def __init__(self, images):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)
        self.images = images
        self.count = len(images)
        self.left_list = []
        self.right_list = []
        self.prepare_lists()

    def prepare_lists(self):
        self.centerIdx = self.count/2 
        self.center_im = self.images[int(self.centerIdx)]
        for i in range(self.count):
            if(i<=self.centerIdx):
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])
        print("Image lists prepared")

    def leftshift(self):
		# self.left_list = reversed(self.left_list)
        a = self.left_list[0]
        i = 0
        for b in self.left_list[1:]:
            i += 1
            print("running left list is %d", i)
            (imageB, imageA) = (a, b)
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)
            # match features between the two images
            M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=2.0)
            # if the match is None, then there aren’t enough matched
            # keypoints to create a panorama
            if M is None:
                return None

            # otherwise, apply a perspective warp to stitch the images
            # together
            (matches, H, status) = M
            print("Homography is : ", H)
            xh = np.linalg.inv(H)
            print("Inverse Homography :", xh)
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
            ds = ds/ds[-1]
            print("final ds=>", ds)
            f1 = np.dot(xh, np.array([0,0,1]))
            f1 = f1/f1[-1]
            xh[0][-1] += abs(f1[0])
            xh[1][-1] += abs(f1[1])
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))
            dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)
            print("image dsize =>", dsize)
            #dsize = ((imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
            tmp = cv2.warpPerspective(a, xh, dsize)
            cv2.imshow("warped", tmp)
            cv2.waitKey()
            tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
            a = tmp
        self.leftImage = tmp

		
    def rightshift(self):
        for each in self.right_list:
            (imageB, imageA) = (self.leftImage, each)
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)
            # match features between the two images
            M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0)
            # if the match is None, then there aren’t enough matched
            # keypoints to create a panorama
            if M is None:
                return None

            # otherwise, apply a perspective warp to stitch the images
            # together
            (matches, H, status) = M
            print("Homography :", H)
            txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            txyz = txyz/txyz[-1]
            dsize = (int(txyz[0])+self.leftImage.shape[1], int(txyz[1])+self.leftImage.shape[0])
            tmp = cv2.warpPerspective(each, H, dsize)
            cv2.imshow("tp", tmp)
            cv2.waitKey()
			# tmp[:self.leftImage.shape[0], :self.leftImage.shape[1]]=self.leftImage
            tmp = self.mix_and_match(self.leftImage, tmp)
            print("tmp shape",tmp.shape)
            print("self.leftimage shape=", self.leftImage.shape)
            self.leftImage = tmp
		# self.showImage('left')



    def mix_and_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]
        print(leftImage[-1,-1])

        t = time.time()
        black_l = np.where(leftImage == np.array([0,0,0]))
        black_wi = np.where(warpedImage == np.array([0,0,0]))
        print(time.time() - t)
        print(black_l[-1])

        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
						# print "BLACK"
						# instead of just putting it with black, 
						# take average of all nearby values and avg it.
                        warpedImage[j,i] = [0, 0, 0]
                    else:
                        if(np.array_equal(warpedImage[j,i],[0,0,0])):
							# print "PIXEL"
                            warpedImage[j,i] = leftImage[j,i]
                        else:
                            if not np.array_equal(leftImage[j,i], [0,0,0]):
                                bw, gw, rw = warpedImage[j,i]
                                bl,gl,rl = leftImage[j,i]
								# b = (bl+bw)/2
								# g = (gl+gw)/2
								# r = (rl+rw)/2
                                warpedImage[j, i] = [bl,gl,rl]
                except:
                    pass
		# cv2.imshow("waRPED mix", warpedImage)
		# cv2.waitKey()
        return warpedImage

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        # if the match is None, then there aren’t enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        
        result = cv2.warpPerspective(imageA, H,
        (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        #result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        for i in range(imageB.shape[0]):
            for j in range(imageB.shape[1]):
                if (imageB[i,j][0] != 0 and imageB[i,j][1] != 0 and imageB[i,j][2]) != 0:
                    result[i,j] = imageB[i,j]
        
        #result = cv2.warpPerspective(imageB, H,
        #(imageB.shape[1] + imageA.shape[1], imageB.shape[0]))
        #result[0:imageA.shape[0], 0:imageA.shape[1]] = imageA

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,status)
            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)
        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
            # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)
            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)
            # convert the keypoints from KeyPoint objects to NumPy
            # arrays
            kps = np.float32([kp.pt for kp in kps])
            # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
    ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe’s ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
            ptsB = np.float32([kpsB[i].pt for (i, _) in matches])
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
        reprojThresh)
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)
        # otherwise, no homograpy could be computed
        return None
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
                ptB = (int(kpsB[trainIdx].pt[0]) + wA, int(kpsB[trainIdx].pt[1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # return the visualization
        return vis


