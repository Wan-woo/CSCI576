# import the necessary packages
from panorama import Stitcher
import argparse
import imutils
import cv2
import os
# construct the argument parse and parse the arguments

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread("images/frame_45.jpg")
imageB = cv2.imread("images/frame_20.jpg")

path = "test3/"
image_list = []
# read files
files = os.listdir(path)
N = len(files)
for i in range(N):
    image_list.append(cv2.imread(path+"/"+"frame_"+str(i+1)+".jpg"))

'''
image_list.append(cv2.imread("images/frame_300.jpg"))
image_list.append(cv2.imread("images/frame_275.jpg"))
#image_list.append(cv2.imread("images/frame_265.jpg"))
#image_list.append(cv2.imread("images/frame_260.jpg"))
#image_list.append(cv2.imread("images/frame_255.jpg"))

image_list.append(cv2.imread("images/frame_250.jpg"))
#image_list.append(cv2.imread("images/frame_235.jpg"))
image_list.append(cv2.imread("images/frame_225.jpg"))


image_list.append(cv2.imread("images/frame_200.jpg"))
image_list.append(cv2.imread("images/frame_175.jpg"))
image_list.append(cv2.imread("images/frame_150.jpg"))
image_list.append(cv2.imread("images/frame_125.jpg"))
image_list.append(cv2.imread("images/frame_100.jpg"))

image_list.append(cv2.imread("images/frame_75.jpg"))
image_list.append(cv2.imread("images/frame_50.jpg"))
image_list.append(cv2.imread("images/frame_20.jpg"))
''''''
'''
'''
image_list.append(cv2.imread("images/S1.jpg"))
image_list.append(cv2.imread("images/S2.jpg"))
image_list.append(cv2.imread("images/S3.jpg"))

image_list.append(cv2.imread("images/S5.jpg"))
image_list.append(cv2.imread("images/S6.jpg"))
'''



# stitch the images together to create a panorama
stitcher = Stitcher(image_list)
'''
result1 = stitcher.stitch([image_list[0], image_list[1]], showMatches=False)
result2 = stitcher.stitch([image_list[2], image_list[3]], showMatches=False)
result3 = stitcher.stitch([image_list[4], image_list[5]], showMatches=False)
result4 = stitcher.stitch([image_list[6], image_list[7]], showMatches=False)
result5 = stitcher.stitch([image_list[8], image_list[9]], showMatches=False)
result6 = stitcher.stitch([image_list[10], image_list[11]], showMatches=False)

image_list.clear()
image_list = [result1, result2, result3, result4, result5, result6]
stitcher.__init__(image_list)
'''
'''
result = stitcher.stitch([image_list[0], image_list[1]], showMatches=False)
for i in range(2, len(image_list)):
    print("current frame is"+str(i))
    result = stitcher.stitch([result, image_list[i]], showMatches=False)

resultA = stitcher.stitch([image_list[0], image_list[1]], showMatches=False)
resultB = stitcher.stitch([image_list[2], image_list[3]], showMatches=False)
result = stitcher.stitch([resultA, resultB], showMatches=False)
'''
#(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

#stitcher.leftshift()
#stitcher.rightshift()
#stitcher.computeHmatrix()

# show the images
#cv2.imshow("Image A", imageA)
#cv2.imshow("Image B", imageB)
#cv2.imshow("Keypoint Matches", vis)
#cv2.imshow("Result", stitcher.leftImage)
#cv2.waitKey(0)
#result = stitcher.reverseStitch([imageA, imageB], showMatches=False)
#cv2.imwrite("Stitched_Panorama.png", stitcher.leftImage)
#cv2.imwrite("result1.png", result)

for i in range(319, 315, -1):
    stitcher.stitch([image_list[i], image_list[268]], showMatches=False, imageIndex=i+1)

#for i in range(80, -1, -1):
#    stitcher.reverseStitch([image_list[i+60], image_list[i]], showMatches=False, imageIndex=i+1)
