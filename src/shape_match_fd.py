#####################################################################
# this file for testing shape matching using fourier descriptor
######################################################################

import numpy as np
import cv2
import time
from PIL import Image

# Main findcontour function
def getContours(img):
    # turn 3 channels into 1
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Dilation make all 6 to form a closed loop
    kernel = np.ones((5, 5), np.uint8)
    imgdilation = cv2.dilate(img, kernel, iterations=2)
    # Must use EXTERNAL outer contours, Must use CHAIN_APPROX_NONE method(not change points)
    imgcontours, contours, hierarchy = cv2.findContours(imgdilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours, hierarchy = cv2.findContours(imgdilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours

# Get complex vector of templete contour
def getTempleteCV(img):
    # Automatically find templete contour
    tpContour = getContours(img)

    templeteComVector = []

    for contour in tpContour:
        x, y, w, h = cv2.boundingRect(contour)
        for point in contour:
            # -x and -y are to make left and upper boundry start from 0
            templeteComVector.append(complex(point[0][0] - x, (point[0][1] - y)))

    return templeteComVector

# Get complex vectors of testees contours
def getSampleCV(img):
    spContours = getContours(img)
    # cv2.drawContours(imgOri, spContours, -1, (0, 0, 255), 1)

    sampleComVectors = []
    sampleContours = []

    for contour in spContours:
        sampleComVector = []
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)

        for point in contour:
            sampleComVector.append(complex(point[0][0] - x, (point[0][1] - y)))
        # sampleComVectors store CV of all testees contours
        sampleComVectors.append(sampleComVector)
        # sampleContours store all testees contours, same order with sampleComVectors
        sampleContours.append(contour)

    return sampleComVectors

# Calculate fourier transform of templete CV
def getempleteFD(templeteComVector):
    return np.fft.fft(templeteComVector)


# Calculate fourier transform of sample CVs
def getsampleFDs(sampleComVectors):
    FDs = []
    for sampleVector in sampleComVectors:
        sampleFD = np.fft.fft(sampleVector)
        FDs.append(sampleFD)

    return FDs


# Make fourier descriptor invariant to rotaition and start point
def rotataionInvariant(fourierDesc):
    for index, value in enumerate(fourierDesc):
        fourierDesc[index] = np.absolute(value)

    return fourierDesc


# Make fourier descriptor invariant to scale
def scaleInvariant(fourierDesc):
    firstVal = fourierDesc[0]

    for index, value in enumerate(fourierDesc):
        fourierDesc[index] = value / firstVal

    return fourierDesc


# Make fourier descriptor invariant to translation
def transInvariant(fourierDesc):
    return fourierDesc[1:len(fourierDesc)]


# Get the lowest X of frequency values from the fourier values.
def getLowFreqFDs(fourierDesc):
    # frequence order returned by np.fft is (0, 0.1, 0.2, 0.3, ...... , -0.3, -0.2, -0.1)
    # Note: in transInvariant(), we already remove first FD(0 frequency)

    return fourierDesc[:5]


# Get the final FD that we want to use to calculate distance
def finalFD(fourierDesc):
    fourierDesc = rotataionInvariant(fourierDesc)
    fourierDesc = scaleInvariant(fourierDesc)
    fourierDesc = transInvariant(fourierDesc)
    fourierDesc = getLowFreqFDs(fourierDesc)

    return fourierDesc

def finalFD_wo_rotinvart(fourierDesc):
    # fourierDesc = rotataionInvariant(fourierDesc)
    fourierDesc = scaleInvariant(fourierDesc)
    fourierDesc = transInvariant(fourierDesc)
    fourierDesc = getLowFreqFDs(fourierDesc)

    return fourierDesc


# Core match function
def match(tpFD, spFDs, flag_rot = False):
    tpFD = finalFD(tpFD)
    # dist store the distance, same order as spContours
    dist = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for spFD in spFDs:
        if flag_rot:
            spFD = finalFD_wo_rotinvart(spFD)
        else:
            spFD = finalFD(spFD)
        # Calculate Euclidean distance between templete and testee
        dist.append(np.linalg.norm(np.array(spFD) - np.array(tpFD)))
        # x, y, w, h = cv2.boundingRect(sampleContours[len(dist) - 1])
        # # Draw distance on image
        # distText = str(round(dist[len(dist) - 1], 2))
        # cv2.putText(imgOri, distText, (x, y - 8), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # # print str(len(dist)) + ": " + str(dist[len(dist)-1])
        # # if distance is less than threshold, it will be good match.
        # if dist[len(dist) - 1] < distThreshold:
        #     cv2.rectangle(imgOri, (x - 5, y - 5), (x + w + 5, y + h + 5), (40, 255, 0), 2)

    return dist

if __name__ == '__main__':
    puzzles = []
    templates = []
    puzzles_pil = []
    templates_pil = []
    # for i in range(4):
    #     img_puzzle = cv2.imread('./../pic/shape_match_test/puzzle_cnt_proj_{}.png'.format(i+1))
    #     puzzles.append(img_puzzle)
    #     img_template = cv2.imread('./../pic/shape_match_test/template_cnt_proj_{}.png'.format(i + 1))
    #     templates.append(img_template)
    puzzles_pil.append(Image.open('./../pic/shape_match_test/puzzle_cnt_proj_{}.png'.format(1)))
    templates_pil.append(Image.open('./../pic/shape_match_test/template_cnt_proj_{}.png'.format(3)))
    puzzles_pil.append(Image.open('./../pic/shape_match_test/puzzle_cnt_proj_{}.png'.format(2)))
    templates_pil.append(Image.open('./../pic/shape_match_test/template_cnt_proj_{}.png'.format(2)))
    puzzles_pil.append(Image.open('./../pic/shape_match_test/puzzle_cnt_proj_{}.png'.format(3)))
    templates_pil.append(Image.open('./../pic/shape_match_test/template_cnt_proj_{}.png'.format(4)))
    puzzles_pil.append(Image.open('./../pic/shape_match_test/puzzle_cnt_proj_{}.png'.format(4)))
    templates_pil.append(Image.open('./../pic/shape_match_test/template_cnt_proj_{}.png'.format(1)))
    puzzles = [np.array(f) for f in puzzles_pil]
    templates = [np.array(f) for f in templates_pil]

    puzzles_hull = []
    templates_hull = []
    for idx, puzzle in enumerate(puzzles):
        cnt = getContours(puzzle)
        mask = np.zeros(puzzle.shape[:2], dtype="uint8")

        ####################
        #  convex hull
        ####################
        hull = cv2.convexHull(cnt[0], False)[:, 0, :]
        mask = cv2.polylines(mask, [hull], isClosed=True, color=255, thickness=3, lineType=8, shift=0)
        puzzles_hull.append(mask)

    for idx, template in enumerate(templates):
        cnt = getContours(template)
        mask = np.zeros(template.shape[:2], dtype="uint8")
        ####################
        #  convex hull
        ####################
        hull = cv2.convexHull(cnt[0], False)[:, 0, :]
        mask = cv2.polylines(mask, [hull], isClosed=True, color=255, thickness=3, lineType=8, shift=0)
        templates_hull.append(mask)

    puzzles_hull_pil = [Image.fromarray(f) for f in puzzles_hull]
    templates_hull_pil = [Image.fromarray(f) for f in templates_hull]

    for idx_p, img_puzzle in enumerate(puzzles_hull):
        ##########################
        # fourier descriptor
        ##########################
        templeteComVector = getTempleteCV(img_puzzle)
        tpFD = getempleteFD(templeteComVector)
        dst_min = 1e10
        template_asso = None
        temp_cnt_asso = None
        template_array = None
        for idx_t, img_template in enumerate(templates_hull):
            sampleContours = getSampleCV(img_template)
            # Get fourider descriptor
            sampleFDs = getsampleFDs(sampleContours)

            # real match function
            dst = match(tpFD, sampleFDs)[0]
            if dst < dst_min:
                dst_min = dst
                template_asso = templates_hull_pil[idx_t]
                temp_cnt_asso = sampleContours
                template_array = img_template

        cv2.namedWindow('puzzle', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('puzzle', img_puzzle)
        cv2.namedWindow('template', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('template', template_array)
        cv2.waitKey(0)
        print("dst_min:{}".format(dst_min))

        dst_min = 1e10
        rotation = 0
        for theta in range(360):
            template_asso_rot = template_asso.rotate(theta)
            template_asso_rot = np.array(template_asso_rot)
            sampleContours_rot = getSampleCV(template_asso_rot)
            sampleFDs_rot = getsampleFDs(sampleContours_rot)
            dst = match(tpFD, sampleFDs_rot, True)[0]
            if dst < dst_min:
                dst_min = dst
                rotation = theta
        #
        # cv2.namedWindow('puzzle', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('puzzle', img_puzzle)
        # cv2.namedWindow('template', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('template', template_asso)
        # cv2.waitKey(0)
        print("rotation:{}".format(-rotation))
        puzzles_hull_pil[idx_p].show()
        rotated_temp = template_asso.rotate(rotation)
        rotated_temp.show()