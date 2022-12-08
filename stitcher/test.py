# import the necessary packages
from panorama import Stitcher
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread("output/frame_100.jpg")
imageB = cv2.imread("output/frame_6.jpg")

image_list = []

image_list.append(cv2.imread("output/frame_590.jpg"))
image_list.append(cv2.imread("output/frame_550.jpg"))
image_list.append(cv2.imread("output/frame_500.jpg"))
#image_list.append(cv2.imread("images/frame_265.jpg"))
#image_list.append(cv2.imread("images/frame_260.jpg"))
#image_list.append(cv2.imread("images/frame_255.jpg"))

image_list.append(cv2.imread("output/frame_450.jpg"))
#image_list.append(cv2.imread("images/frame_235.jpg"))
image_list.append(cv2.imread("output/frame_400.jpg"))

'''
image_list.append(cv2.imread("test3/frame_290.jpg"))
image_list.append(cv2.imread("test3/frame_260.jpg"))
image_list.append(cv2.imread("test3/frame_200.jpg"))
image_list.append(cv2.imread("test3/frame_170.jpg"))
image_list.append(cv2.imread("test3/frame_140.jpg"))

image_list.append(cv2.imread("test3/frame_110.jpg"))
image_list.append(cv2.imread("test3/frame_80.jpg"))
image_list.append(cv2.imread("test3/frame_50.jpg"))
'''
''''''



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
result = stitcher.stitch([imageA, imageB], showMatches=False)
#cv2.imwrite("Stitched_Panorama.png", stitcher.leftImage)
cv2.imwrite("result_test3_6_100.png", result)
