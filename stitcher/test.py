# import the necessary packages
from panorama import Stitcher
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread("images/frame_100.jpg")
imageB = cv2.imread("images/frame_20.jpg")

image_list = []
# from left to right
image_left = []
iamge_right = []
image_list.append(cv2.imread("images/frame_345.jpg"))
image_list.append(cv2.imread("images/frame_300.jpg"))
image_list.append(cv2.imread("images/frame_250.jpg"))

image_list.append(cv2.imread("images/frame_200.jpg"))
image_list.append(cv2.imread("images/frame_150.jpg"))
image_list.append(cv2.imread("images/frame_100.jpg"))
image_list.append(cv2.imread("images/frame_20.jpg"))

# stitch the images together to create a panorama
stitcher = Stitcher()
'''
result = stitcher.stitch([image_list[0], image_list[1]], showMatches=False)
for i in range(2, len(image_list)):
    print("current frame is"+str(i))
    result = stitcher.stitch([result, image_list[i]], showMatches=False)
'''
resultA = stitcher.stitch([image_list[0], image_list[1]], showMatches=False)
resultB = stitcher.stitch([image_list[2], image_list[3]], showMatches=False)
result = stitcher.stitch([resultA, resultB], showMatches=False)

#(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)



# show the images
#cv2.imshow("Image A", imageA)
#cv2.imshow("Image B", imageB)
#cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
