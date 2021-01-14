import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import cv2
import time
from PIL import Image

from shape_match_fd import getContours, getTempleteCV, getSampleCV, getempleteFD, getsampleFDs, \
    rotataionInvariant, scaleInvariant, transInvariant, getLowFreqFDs, \
    finalFD, finalFD_wo_rotinvart, match

def eig_getter(img):
   y, x = np.nonzero(img)
   x = x - np.mean(x)
   y = y - np.mean(y)
   coords = np.vstack([x, y])
   cov = np.cov(coords)
   evals, evecs = np.linalg.eig(cov)
   sort_indices = np.argsort(evals)[::-1]
   x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
   x_v2, y_v2 = evecs[:, sort_indices[1]]
   return x, y, x_v1, y_v1, x_v2, y_v2

def find_longest_line(cnt):
    '''
    :param cnt: contour: a set of points
    :return: v_x, v_y, x, y coordinate of a point and slope
    '''
    pxl1 = None
    pxl2 = None
    dst_max = 0.0
    start = time.time()
    for idx, pxl_1 in enumerate(cnt):
        if idx == cnt.shape[0]:
            break
        for pxl_2 in cnt[idx+1 : -1]:
            dst = np.sqrt(np.sum((pxl_1 - pxl_2) ** 2))
            if dst > dst_max:
                dst_max = dst
                pxl1 = pxl_1
                pxl2 = pxl_2
    end = time.time()
    print("time cost:{}".format(end-start))
    v_x = pxl2[0, 0] - pxl1[0, 0]
    v_y = pxl2[0, 1] - pxl1[0, 1]
    x = pxl1[0, 0]
    y = pxl1[0, 1]

    return v_x, v_y, x, y

def rotatePoint(centerPoint,point,angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    # angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    temp_point = (temp_point[0]*np.cos(angle)-temp_point[1]*np.sin(angle), temp_point[0]*np.sin(angle)+temp_point[1]*np.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point

def rotatePoints(centerPoint, points, angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    # angle = math.radians(angle)
    temp_points = points - centerPoint
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    temp_points = rot_mat.dot(temp_points.T)
    temp_points = temp_points.T + centerPoint

    return temp_points

if __name__ == '__main__':
    puzzles = []
    templates = []

    puzzles.append(cv2.imread('./../pic/shape_match_test/puzzle_cnt_proj_{}.png'.format(1)))
    templates.append(cv2.imread('./../pic/shape_match_test/template_cnt_proj_{}.png'.format(3)))
    puzzles.append(cv2.imread('./../pic/shape_match_test/puzzle_cnt_proj_{}.png'.format(2)))
    templates.append(cv2.imread('./../pic/shape_match_test/template_cnt_proj_{}.png'.format(2)))
    puzzles.append(cv2.imread('./../pic/shape_match_test/puzzle_cnt_proj_{}.png'.format(3)))
    templates.append(cv2.imread('./../pic/shape_match_test/template_cnt_proj_{}.png'.format(4)))
    puzzles.append(cv2.imread('./../pic/shape_match_test/puzzle_cnt_proj_{}.png'.format(4)))
    templates.append(cv2.imread('./../pic/shape_match_test/template_cnt_proj_{}.png'.format(1)))

    puzzles_hull = []
    templates_hull = []
    for idx, puzzle in enumerate(puzzles):
        cnt = getContours(puzzle)
        mask = np.zeros(puzzle.shape[:2], dtype="uint8")

        ####################
        #  convex hull
        ####################
        hull = cv2.convexHull(cnt[0], False)[:, 0, :]
        mask = cv2.polylines(mask, [hull], isClosed=True, color=255, thickness=2, lineType=8, shift=0)
        puzzles_hull.append(mask)

    for idx, template in enumerate(templates):
        cnt = getContours(template)
        mask = np.zeros(template.shape[:2], dtype="uint8")
        ####################
        #  convex hull
        ####################
        hull = cv2.convexHull(cnt[0], False)[:, 0, :]
        mask = cv2.polylines(mask, [hull], isClosed=True, color=255, thickness=2, lineType=8, shift=0)
        templates_hull.append(mask)

    for idx in range(4):
        puzzle = puzzles_hull[idx]
        # puzzle = cv2.cvtColor(puzzle, cv2.COLOR_BGR2GRAY)
        template = templates_hull[idx]
        # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        #####################
        #   PCA
        #####################
    for idx, puzzle in enumerate(puzzles_hull):
        # h_p, w_p = puzzle.shape
        mat_p = np.argwhere(puzzle != 0)
        mat_p[:, [0, 1]] = mat_p[:, [1, 0]]
        mat_p = np.array(mat_p).astype(np.float32)

        m_p, e_p = cv2.PCACompute(mat_p, mean=np.array([]))

        # main axis
        theta_p = np.arctan2(e_p[0][1], e_p[0][0])

        idx_y, idx_x = np.where(puzzle != 0)
        points = np.empty((idx_x.shape[0], 2))
        points[:, 0] = idx_x
        points[:, 1] = idx_y

        template = templates_hull[idx]
        mat_t = np.argwhere(template != 0)
        mat_t[:, [0, 1]] = mat_t[:, [1, 0]]
        mat_t = np.array(mat_t).astype(np.float32)

        m_t, e_t = cv2.PCACompute(mat_t, mean=np.array([]))

        theta_t_pri = np.arctan2(e_t[0][1], e_t[0][0])
        theta_t_sec = np.arctan2(e_t[1][1], e_t[1][0])

        dst_max = 0
        theta_t = 0
        for theta in [theta_t_pri, theta_t_pri - np.pi, theta_t_sec, theta_t_sec - np.pi]:
            angle = theta - theta_p
            # print(angle)
            points_rot = rotatePoints(m_p[0], points, angle)
            t = m_p[0] - m_t[0]
            points_rot = points_rot - t
            points_rot = points_rot.astype(int)
            points_rot[:, [0, 1]] = points_rot[:, [1, 0]]

            mask = np.zeros(puzzle.shape[:2], dtype="uint8")
            for point_rot in points_rot:
                mask[point_rot[0], point_rot[1]] = 255

            # Perform match operations.
            res = cv2.matchTemplate(template, mask, cv2.TM_CCOEFF_NORMED)

            dst = np.amax(res)

            # print(dst)
            # cv2.namedWindow('rot', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('rot', mask)
            # cv2.namedWindow('template', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('template', template)
            # cv2.waitKey(0)

            if dst > dst_max:
                dst_max = dst
                theta_t = theta

        print("final angle:{}".format(theta_t - theta_p))
        # cv2.namedWindow('ori', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('ori', puzzle)
        points_rot = rotatePoints(m_p[0], points, theta_t-theta_p)
        points_rot = points_rot.astype(int)
        points_rot[:, [0, 1]] = points_rot[:, [1, 0]]
        mask = np.zeros(puzzle.shape[:2], dtype="uint8")
        for point_rot in points_rot:
            mask[point_rot[0], point_rot[1]] = 255
        # cv2.namedWindow('rot', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('rot', mask)
        # cv2.namedWindow('template', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('template', template)
        # cv2.waitKey(0)


        #####################
        #   eig
        #####################
        # x_puzzle, y_puzzle, x_v1_puzzle, y_v1_puzzle, x_v2_puzzle, y_v2_puzzle = eig_getter(puzzle)
        # x_template, y_template, x_v1_template, y_v1_template, x_v2_template, y_v2_template = eig_getter(template)
        #
        # scale = 20
        # # plot puzzle image
        # plt.plot([x_v1_puzzle * -scale * 2, x_v1_puzzle * scale * 2],
        #          [y_v1_puzzle * -scale * 2, y_v1_puzzle * scale * 2], color='red')
        # plt.plot([x_v2_puzzle * -scale, x_v2_puzzle * scale],
        #          [y_v2_puzzle * -scale, y_v2_puzzle * scale], color='blue')
        # plt.plot(x_puzzle, y_puzzle, 'k.')
        # plt.axis('equal')
        # plt.gca().invert_yaxis()
        #
        # plt.plot([x_v1_template * -scale * 2, x_v1_template * scale * 2],
        #          [y_v1_template * -scale * 2, y_v1_template * scale * 2], color='black')
        # plt.plot([x_v2_template * -scale, x_v2_template * scale],
        #          [y_v2_template * -scale, y_v2_template * scale], color='green')
        # plt.plot(x_template, y_template, 'm.')
        # plt.axis('equal')
        # plt.gca().invert_yaxis()
        #
        # theta_puzzle = np.arctan2(y_v1_puzzle, x_v1_puzzle)
        # theta_template = np.arctan2(y_v1_template, x_v1_template)
        # print('actual', theta_puzzle)
        # print('template', theta_template)
        # theta = theta_template - theta_puzzle
        # print('Theta diff', theta)
        #
        # # sanity check: if eigvecs similar DONT Rotate:
        # # apply rotation first
        # # rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
        # #                           [np.sin(theta), np.cos(theta)]])
        # # coords = np.vstack([x_puzzle, y_puzzle])
        # # transformed_mat = rotation_mat * coords
        # #
        # # # plot the transformed blob
        # # x_transformed, y_transformed = transformed_mat.A
        # #
        # # template_coords = []
        # # transform_coords = []
        # # # idx = 0
        # # # make the two lists
        # # for idx, x in enumerate(x_template):
        # #     template_coords = template_coords + [[x, y_template[idx]]]
        # #     transform_coords = transform_coords + [[x_transformed[idx], y_transformed[idx]]]
        # #     # idx += 1
        # # # number of matching points for normal case
        # # idx1 = 0
        # # final_sum = 0
        # #
        # # # compare the two lists pairwise
        # # for pair in template_coords:
        # #     for pair2 in transform_coords:
        # #         if ((pair[0] - pair2[0]) ** 2 + (pair[1] - pair2[1]) ** 2) ** 0.5 < threshold:
        # #             idx1 += 1
        # # print('found number of matching points', idx1)
        # #
        # # rotation_mat_new = np.matrix([[np.cos(np.pi), -np.sin(np.pi)],
        # #                               [np.sin(np.pi), np.cos(np.pi)]])
        # # transformed_mat_new = rotation_mat_new * transformed_mat
        # # x_transformed_new, y_transformed_new = transformed_mat_new.A
        # # # number of matching points for inverted (180 degrees) case
        # # idx2 = 0
        # # transformed_coords = []
        # # for x in x_transformed_new:
        # #     transformed_coords = transformed_coords + [[x_transformed_new[idx2], y_transformed_new[idx2]]]
        # #     idx2 += 1
        # # idx2 = 0
        # # for pair in template_coords:
        # #     for pair2 in transformed_coords:
        # #         if ((pair[0] - pair2[0]) ** 2 + (pair[1] - pair2[1]) ** 2) ** 0.5 < threshold:
        # #             idx2 += 1
        # # print('found number of matching points in 180 case', idx2)
        # #
        # # # if idx2> idx1, the inverted case is correct
        # # if idx2 > idx1:
        # #     plt.plot(x_transformed_new, y_transformed_new, 'g.')
        # #     angle = np.pi - theta
        # # # else stick with the normal case.
        # # else:
        # #     plt.plot(x_transformed, y_transformed, 'g.')
        # #     angle = -theta
        #
        # print('angle to be rotated counter-clockwise by motor 5 is ', theta / np.pi * 180.)
        # plt.plot(x_template, y_template, 'r.')
        #
        # plt.show()

        ############################
        # cv2
        ############################
        # rows, cols = puzzle.shape[:2]
        #
        # kernel = np.ones((5, 5), np.uint8)
        # puzzle_dilated = cv2.dilate(puzzle, kernel, iterations=2)
        # template_dilated = cv2.dilate(template, kernel, iterations=2)
        # contours_p, hierarchy_p = cv2.findContours(puzzle_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours_t, hierarchy_t = cv2.findContours(template_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #
        # # fit a line/ longest line
        # # [vx_p, vy_p, x_p, y_p] = cv2.fitLine(contours_p[0], cv2.DIST_L2, 0, 0.01, 0.01)
        # [vx_p, vy_p, x_p, y_p] = find_longest_line(contours_p[0])
        # theta_p = np.arctan2(vy_p, vx_p)
        # lefty_p = int((-x_p * vy_p / vx_p) + y_p)
        # righty_p = int(((cols - x_p) * vy_p / vx_p) + y_p)
        # img_p = cv2.line(puzzle, (cols - 1, righty_p), (0, lefty_p), 255, 2)
        #
        # # [vx_t, vy_t, x_t, y_t] = cv2.fitLine(contours_t[0], cv2.DIST_L2, 0, 0.01, 0.01)
        # [vx_t, vy_t, x_t, y_t] = find_longest_line(contours_t[0])
        # theta_t = np.arctan2(vy_t, vx_t)
        # lefty_t = int((-x_t * vy_t / vx_t) + y_t)
        # righty_t = int(((cols - x_t) * vy_t / vx_t) + y_t)
        # img_t = cv2.line(template, (cols - 1, righty_t), (0, lefty_t), 255, 2)
        #
        # cv2.namedWindow('p', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('p', img_p)
        # cv2.namedWindow('t', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('t', img_t)
        # cv2.waitKey(0)
