import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import imutils
from scipy import ndimage

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

align_to = rs.stream.color
align = rs.align(align_to)

# Start streaming
pipeline.start(config)

lower_hsv = np.array([15, 95, 0])
upper_hsv = np.array([25, 115, 255])
lower_h = np.array([15, 0, 0])
upper_h = np.array([25, 255, 255])
lower_s = np.array([0, 0, 0])
upper_s = np.array([179, 25, 255])
lower_v = np.array([0, 0, 0])
upper_v = np.array([179, 255, 50])

# Realsense D435 depth camera intrinsic matrix
cameraMatrix = np.array([[613.8052368164062, 0.0, 328.09918212890625],
                         [0.0, 613.8528442382812, 242.4539337158203],
                         [0.0, 0.0, 1.0]])
distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# default transformation matrix from camera to aruco tag
default_M_CL = np.array([[8.05203923e-04,  -9.98258274e-01,  5.89895796e-02, 2.11182116e-02],
                         [-7.25197650e-01, -4.11996435e-02, -6.87307033e-01, 1.19383476e-01],
                         [6.88540282e-01,  -4.22256822e-02, -7.23967728e-01, 5.68361874e-01],
                         [0.00000000e+00,   0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

M_CpL = np.array([[ 0.0,  -1.0,  0.0, 0.0],
                  [-1.0,   0.0,  0.0, 0.0],
                  [ 0.0,   0.0, -1.0, 0.5],
                  [ 0.0,   0.0,  0.0, 1.0]])

# transformation matrix from base to aruco tag
M_BL = np.array([[1., 0., 0.,  0.38945],
                 [0., 1., 0.,  0.29145],
                 [0., 0., 1.,  0.00000],
                 [0., 0., 0.,  1.00000]])

pointThreshold_min = 20
pointThreshold_max = 200

def image_perspective_tf(img_src, M_CL, M_CpL):
    '''
    project pixels in one image plane to another image plane

    Arguments:
    img_src: source image
    M_CL: transformation matrix from old camera frame to aruco tag
    M_CpL: transformation matrix from new camera frame to aruco tag

    Return:
    img_dst: projected image
    '''
    M = np.linalg.inv(cameraMatrix).dot(M_CL).dot(np.linalg.inv(M_CpL)).dot(cameraMatrix)
    H, W = img_src.shape

    img_dst = cv2.warpPerspective(img, M, (H, W))

    return img_dst

def table_range(mask):
    '''
    Find the range of table and crop image

    Arguments:
    mask: gray image

    Return:
    mask_table: cropped mask of table
    pxl_min: tuple, location of top-left pixel
    pxl_max: tuple, location of bottom-right pixel
    '''
    index_y, index_x = np.where(mask == 0)
    y_min = np.unique(index_y)[0]
    while True:
        # if y_min > 479:
        #     break

        # if np.where(index_y == y_min)[0].size > 320:
        #     break
        if np.count_nonzero(mask[y_min, :]) == 0:
            break
        y_min += 1

    index_y_min = np.where(index_y == y_min)[0]
    index_x_y_min = index_x[index_y_min]
    x_min = np.amin(index_x_y_min)

    pxl_min = (x_min, y_min)

    dst = np.sqrt(index_x ** 2 + index_y ** 2)
    index_max = np.argmax(dst)

    crop_x_max = index_x[index_max]
    crop_y_max = index_y[index_max]
    pxl_max = (crop_x_max, crop_y_max)

    mask_table = mask[pxl_min[1]:pxl_max[1], pxl_min[0]:pxl_max[0]]

    return mask_table, pxl_min, pxl_max

def aruco_tag_filter(mask, M_CL):
    '''
    Filter the aruco tag out of detected mask according transformation from camera to aruco_tag.

    Arguments:
    mask: gray image
    M_CL: transformation from camera to aruco_tag

    Return:

    '''
    filtered_mask = mask.copy()

    q1_L = np.array([-0.038, -0.038, 0, 1])
    q2_L = np.array([ 0.038, -0.038, 0, 1])
    q3_L = np.array([ 0.038,  0.038, 0, 1])
    q4_L = np.array([-0.038,  0.038, 0, 1])

    q1_C = M_CL.dot(q1_L)
    q2_C = M_CL.dot(q2_L)
    q3_C = M_CL.dot(q3_L)
    q4_C = M_CL.dot(q4_L)

    p1 = cameraMatrix.dot(q1_C[:-1])
    p1[:-1] = p1[:-1] / p1[-1]
    p2 = cameraMatrix.dot(q2_C[:-1])
    p2[:-1] = p2[:-1] / p2[-1]
    p3 = cameraMatrix.dot(q3_C[:-1])
    p3[:-1] = p3[:-1] / p3[-1]
    p4 = cameraMatrix.dot(q4_C[:-1])
    p4[:-1] = p4[:-1] / p4[-1]

    x_min = np.floor(np.amin(np.array([p1[0], p2[0], p3[0], p4[0]])))
    y_min = np.floor(np.amin(np.array([p1[1], p2[1], p3[1], p4[1]])))
    x_max = np.ceil(np.amax(np.array([p1[0], p2[0], p3[0], p4[0]])))
    y_max = np.ceil(np.amax(np.array([p1[1], p2[1], p3[1], p4[1]])))

    filtered_mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 0

    return filtered_mask

def camera2camera_prime(pixles, M_CL, t_x, t_y):
    '''
    project pixels in one image plane to another image plane

    Arguments:
    pixels: (Nx3), N is the number of white pixel in the old image plane, 3 is (u*z, v*z, z)
    M_CL: transformation matrix from old camera frame to aruco tag
    M_CpL: transformation matrix from new camera frame to aruco tag

    Return:
    pixels_pro: (Nx2), N is the number of white pixel in the new image plane, 2 is (u, v)
    '''
    M_CpL = np.array([[ 0., -1.,  0.,  t_x],
                      [-1.,  0.,  0.,  t_y],
                      [ 0.,  0., -1.,  0.50000],
                      [ 0.,  0.,  0.,  1.00000]])
    # transform pixel on the old image plane to 3D points in the old camera frame
    q_C = np.dot(np.linalg.inv(cameraMatrix), pixles.T)
    # insert 1 to the last element
    q_C = np.insert(q_C, 3, 1, axis=0)
    # transform 3d points from old camera frame to aruco frame
    q_L = np.linalg.inv(M_CL).dot(q_C)
    # transform 3d points from aruco frame to new camera frame
    q_Cp = M_CpL.dot(q_L)
    q_Cp = q_Cp[:-1, :]
    # project 3d points into new image plane
    pixles_Cp = cameraMatrix.dot(q_Cp).T
    pixles_Cp[:, 0] /= pixles_Cp[:, 2]
    pixles_Cp[:, 1] /= pixles_Cp[:, 2]
    pixles_Cp[:, 2] /= pixles_Cp[:, 2]

    return pixles_Cp[:, :-1].astype(int)

#####################################################
# Function: get the T matrix from camera to aruco tag
#####################################################
def get_M_CL(gray, image_init, visualize=False):
    # parameters
    markerLength_CL = 0.076
    # aruco_dict_CL = aruco.Dictionary_get(aruco.DICT_5X5_250)
    aruco_dict_CL = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    corners_CL, ids_CL, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict_CL, parameters=parameters)

    # for the first frame, it may contain nothing
    if ids_CL is None:
        return default_M_CL

    rvec_CL, tvec_CL, _objPoints_CL = aruco.estimatePoseSingleMarkers(corners_CL[0], markerLength_CL,
                                                                      cameraMatrix, distCoeffs)
    dst_CL, jacobian_CL = cv2.Rodrigues(rvec_CL)
    M_CL = np.zeros((4, 4))
    M_CL[:3, :3] = dst_CL
    M_CL[:3, 3] = tvec_CL
    M_CL[3, :] = np.array([0, 0, 0, 1])

    # print(M_CL)

    if visualize:
        # print('aruco is located at mean position (%d, %d)' %(mean_x ,mean_y))
        aruco.drawAxis(image_init, cameraMatrix, distCoeffs, rvec_CL, tvec_CL, markerLength_CL)
    return M_CL

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        # no_zero_pxl = depth_image.shape[0] * depth_image.shape[1] - np.count_nonzero(depth_image)
        color_image = np.asanyarray(color_frame.get_data())
        gray = color_image.astype(np.uint8)

        # (1)h(2)s(3)v threshold
        original = color_image.copy()
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_h, upper_h)
        mask2 = cv2.inRange(hsv, lower_s, upper_s)
        mask3 = cv2.inRange(hsv, lower_v, upper_v)
        mask2 = 255 - mask2
        # mask3 = 255 - mask3
        mask = cv2.bitwise_and(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)

        # AR tag detection and filter it out in the above mask
        M_CL = get_M_CL(gray, color_image, False)
        mask_wo_arucotag = aruco_tag_filter(mask, M_CL)

        # crop mask only contains table
        mask_table, pxl_min, pxl_max = table_range(mask_wo_arucotag)

        # crop depth raw image
        depth_raw_crop = depth_image[pxl_min[1]: pxl_max[1], pxl_min[0]: pxl_max[0]]
        # no_zero_pxl = depth_raw_crop.shape[0] * depth_raw_crop.shape[1] - np.count_nonzero(depth_raw_crop)
        # if no_zero_pxl > 40000:
        #     continue

        # smooth the mask
        kernel = np.ones((5, 5), np.uint8)
        # cv2 openning
        # mask_smoothed = cv2.morphologyEx(mask_table, cv2.MORPH_OPEN, kernel)
        # cv2 closing
        mask_smoothed = cv2.morphologyEx(mask_table, cv2.MORPH_CLOSE, kernel)
        # cv2 erode
        # mask_smoothed = cv2.erode(mask_table, kernel, iterations=1)

        # Show images
        cv2.namedWindow('RGB image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB image', color_image)
        cv2.namedWindow('detected mask', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('detected mask', mask_smoothed)
        cv2.waitKey(1)

        # find the contours for each puzzle piece
        cnts = cv2.findContours(mask_smoothed.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        puzzle_cnts = []
        plate_cnt = []
        area_max = 0.0
        for idx, cnt in enumerate(cnts):

            if len(cnt) > pointThreshold_min:
                # x = 0
                # y = 0
                # for z in range(cnt.shape[0]):
                #     # sum up all the x,y vals
                #     row = cnt[z][0]
                #     x = x + row[0]
                #     y = y + row[1]
                # mean_x = int(x / cnt.shape[0])
                # mean_y = int(y / cnt.shape[0])
                # print("mean pixel of {} cnt is ({}, {})".format(idx, mean_x, mean_y))
                # print("total number of cnt:{}".format(len(cnt)))
                area = (np.amax(cnt[:, 0, 0]) - np.amin(cnt[:, 0, 0])) * (np.amax(cnt[:, 0, 1]) - np.amin(cnt[:, 0, 1]))
                if area > area_max:
                    area_max = area
                    if len(plate_cnt) == 0:
                        plate_cnt.append(cnt)
                    else:
                        puzzle_cnts.append(plate_cnt[0])
                        plate_cnt = []
                        plate_cnt.append(cnt)
                else:
                    puzzle_cnts.append(cnt)

        print("size of detected cnts: {}".format(len(puzzle_cnts) + 1))
        if len(puzzle_cnts) == 0:
            continue

        # find the split line between puzzle pieces and puzzle plate
        y_min = 1000
        for cnt in puzzle_cnts:
            temp = cnt[:, :, 1]
            if np.amin(cnt[:, :, 1]) < y_min:
                y_min = np.amin(cnt[:, :, 1])

        plate_mask = mask_smoothed.copy()
        plate_mask[y_min:, :] = 0
        print("split line: {}".format(y_min))

        mask_p = np.zeros(mask_smoothed.shape[:2], dtype="uint8")
        cv2.polylines(mask_p, puzzle_cnts, True, (255, 255, 255), thickness=3)
        mask_p = cv2.medianBlur(mask_p, 5)
        # cv2.imwrite('./../../pic/jigsaw_puzzle/demo/puzzle_cnts.png', mask_p)
        # cv2.drawContours(mask_p, puzzle_cnts, -1, (255, 255, 255), 3)
        cv2.namedWindow('puzzle cnts', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('puzzle cnts', mask_p)
        cv2.waitKey(1)

        cnts_template = cv2.findContours(plate_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_template = imutils.grab_contours(cnts_template)
        template_cnts = []

        mask_t_pri = np.zeros(plate_mask.shape[:2], dtype="uint8")
        cv2.polylines(mask_t_pri, cnts_template, True, (255, 255, 255), thickness=3)
        mask_t_pri = cv2.medianBlur(mask_t_pri, 5)
        # cv2.imwrite('./../../pic/jigsaw_puzzle/demo/puzzle_cnts.png', mask_p)
        # cv2.drawContours(mask_p, puzzle_cnts, -1, (255, 255, 255), 3)
        cv2.namedWindow('template pri cnts', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('template pri cnts', mask_t_pri)
        cv2.waitKey(1)

        # flip the plate and template puzzle
        if len(cnts_template) != 0:
            H, W = plate_mask.shape
            for h in range(H):
                for w in range(W):
                    if cv2.pointPolygonTest(cnts_template[0], (w, h), False) == 1:
                        plate_mask[h, w] = 255 - plate_mask[h, w]
                    else:
                        plate_mask[h, w] = 0

            cnts_template = cv2.findContours(plate_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts_template = imutils.grab_contours(cnts_template)

            for idx, cnt in enumerate(cnts_template):
                if pointThreshold_min < len(cnt):
                    template_cnts.append(cnt)

            mask_t = np.zeros(mask_smoothed.shape[:2], dtype="uint8")
            cv2.polylines(mask_t, template_cnts, True, (255, 255, 255), thickness=3)
            mask_t = cv2.medianBlur(mask_t, 5)
            # cv2.imwrite('./../../pic/jigsaw_puzzle/demo/template_cnts.png', mask_t)
            # cv2.drawContours(mask_t, template_cnts, -1, (255, 255, 255), 3)
            cv2.namedWindow('template cnts', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('template cnts', mask_t)
            cv2.waitKey(1)

        # skip if number of detected puzzle and template doesn't match
        if len(template_cnts) != len(puzzle_cnts):
            continue

        ####################
        # projection
        ####################
        # collect (u*z, v*z, z) for puzzle
        pxls_puzzle = []
        t_p_xs = []
        t_p_ys = []
        for cnt in puzzle_cnts:
            no_pxl = cnt.shape[0]
            mean_pxl_u = 0
            mean_pxl_v = 0
            pxl_puzzle = np.empty((no_pxl, 3))
            for idx, pxl in enumerate(cnt):
                mean_pxl_u += pxl[0, 0]
                mean_pxl_v += pxl[0, 1]

                depth = depth_raw_crop[pxl[0, 1], pxl[0, 0]] / 1000.
                # hole filling by finding the nearest pixel whose depth is registered
                # average within the circle
                # if depth == 0 or depth > 1.:
                #     radius = 1
                #     depth_filled = 0
                #     while depth_filled == 0:
                #         # count number of pixel satisfying within the circle and depth is registered
                #         count = 0
                #         for pxl_search in cnt:
                #             if (pxl_search[0, 0] - pxl[0, 0]) ** 2 + (pxl_search[0, 1] - pxl[0, 1]) < radius ** 2:
                #                 if depth_raw_crop[pxl_search[0, 1], pxl_search[0, 0]] / 1000. != 0:
                #                     depth_filled += (depth_raw_crop[pxl_search[0, 1], pxl_search[0, 0]] / 1000.)
                #                     count += 1
                #         radius += 1
                #         if count != 0:
                #             depth_filled /= count
                #
                #     depth = depth_filled

                # find the nearest pixel
                if depth == 0 or depth > 1.:
                    dst_min = 1000.0
                    for pxl_search in cnt:
                        if depth_raw_crop[pxl_search[0, 1], pxl_search[0, 0]] / 1000. != 0:
                            dst = np.sqrt((pxl_search[0, 0] - pxl[0, 0]) ** 2 + (pxl_search[0, 1] - pxl[0, 1]) ** 2)
                            if dst < dst_min:
                                dst_min = dst
                                depth = depth_raw_crop[pxl_search[0, 1], pxl_search[0, 0]] / 1000.

                pxl_puzzle[idx, :] = np.array([pxl[0, 0] * depth, pxl[0, 1] * depth, depth])

            pxls_puzzle.append(pxl_puzzle)
            mean_pxl_u /= no_pxl
            mean_pxl_v /= no_pxl
            depth = depth_raw_crop[int(mean_pxl_v), int(mean_pxl_u)] / 1000.
            mean_pxl_x = (mean_pxl_u * depth - cameraMatrix[0, 2] * depth) / cameraMatrix[0, 0]
            mean_pxl_y = (mean_pxl_v * depth - cameraMatrix[1, 2] * depth) / cameraMatrix[1, 1]
            q_C = np.array([mean_pxl_x, mean_pxl_y, depth, 1])
            q_L = np.linalg.inv(M_CL).dot(q_C)
            t_p_xs.append(q_L[1])
            t_p_ys.append(q_L[0])

        # collect (u*z, v*z, z) for template
        pxls_template = []
        t_t_xs = []
        t_t_ys = []
        for cnt in template_cnts:
            no_pxl = cnt.shape[0]
            mean_pxl_u = 0
            mean_pxl_v = 0
            pxl_puzzle = np.empty((no_pxl, 3))
            for idx, pxl in enumerate(cnt):
                mean_pxl_u += pxl[0, 0]
                mean_pxl_v += pxl[0, 1]

                depth = depth_raw_crop[pxl[0, 1], pxl[0, 0]] / 1000.

                # hole filling by finding the nearest pixel whose depth is registered
                # average within the circle
                # if depth == 0 or depth > 1.:
                #     radius = 1
                #     depth_filled = 0
                #     while depth_filled == 0:
                #         # count number of pixel satisfying within the circle and depth is registered
                #         count = 0
                #         for pxl_search in cnt:
                #             if (pxl_search[0, 0] - pxl[0, 0]) ** 2 + (pxl_search[0, 1] - pxl[0, 1]) < radius ** 2:
                #                 if depth_raw_crop[pxl_search[0, 1], pxl_search[0, 0]] / 1000. != 0:
                #                     depth_filled += (depth_raw_crop[pxl_search[0, 1], pxl_search[0, 0]] / 1000.)
                #                     count += 1
                #         radius += 1
                #         if count != 0:
                #             depth_filled /= count
                #
                #     depth = depth_filled

                # find the nearest pixel
                if depth == 0 or depth > 1.:
                    dst_min = 1000.0
                    for pxl_search in cnt:
                        if depth_raw_crop[pxl_search[0, 1], pxl_search[0, 0]] / 1000. != 0:
                            dst = np.sqrt((pxl_search[0, 0] - pxl[0, 0]) ** 2 + (pxl_search[0, 1] - pxl[0, 1]) ** 2)
                            if dst < dst_min:
                                dst_min = dst
                                depth = depth_raw_crop[pxl_search[0, 1], pxl_search[0, 0]] / 1000.

                pxl_puzzle[idx, :] = np.array([pxl[0, 0] * depth, pxl[0, 1] * depth, depth])

            pxls_template.append(pxl_puzzle)
            mean_pxl_u /= no_pxl
            mean_pxl_v /= no_pxl
            depth = depth_raw_crop[int(mean_pxl_v), int(mean_pxl_u)] / 1000.
            mean_pxl_x = (mean_pxl_u * depth - cameraMatrix[0, 2] * depth) / cameraMatrix[0, 0]
            mean_pxl_y = (mean_pxl_v * depth - cameraMatrix[1, 2] * depth) / cameraMatrix[1, 1]
            q_C = np.array([mean_pxl_x, mean_pxl_y, depth, 1])
            q_L = np.linalg.inv(M_CL).dot(q_C)
            t_t_xs.append(q_L[1])
            t_t_ys.append(q_L[0])

        # do projection for puzzle
        pxls_puzzle_proj = []
        for idx, pxl_puzzle in enumerate(pxls_puzzle):
            pxl_puzzle_proj = camera2camera_prime(pxl_puzzle, M_CL, t_p_xs[idx], t_p_ys[idx])
            pxls_puzzle_proj.append(pxl_puzzle_proj)
        # do projection for template
        pxls_template_proj = []
        for idx, pxl_template in enumerate(pxls_template):
            pxl_template_proj = camera2camera_prime(pxl_template, M_CL, t_t_xs[idx], t_t_ys[idx])
            pxls_template_proj.append(pxl_template_proj)

        # compute the center of contour
        for pxl_puzzle_proj in pxls_puzzle_proj:
            mean_x = 0
            mean_y = 0
            for pxl in pxl_puzzle_proj:
                mean_x += pxl[0]
                mean_y += pxl[1]

            mean_x /= pxl_puzzle_proj.shape[0]
            mean_y /= pxl_puzzle_proj.shape[0]
            print("mean x and y: {} and {}".format(mean_x, mean_y))

        for pxl_template_proj in pxls_template_proj:
            mean_x = 0
            mean_y = 0
            for pxl in pxl_template_proj:
                mean_x += pxl[0]
                mean_y += pxl[1]

            mean_x /= pxl_template_proj.shape[0]
            mean_y /= pxl_template_proj.shape[0]
            print("mean x and y: {} and {}".format(mean_x, mean_y))

        # if len(pxls_puzzle_proj) != 0:
        #     for idx, cnt in enumerate(pxls_puzzle_proj):
        #         mask_save = np.zeros(mask_table.shape[:2], dtype="uint8")
        #         cv2.polylines(mask_save, [cnt], True, (255, 255, 255), thickness=3)
        #         # cv2.drawContours(mask_save, [cnt], -1, (255, 255, 255), 3)
        #         mask_save = cv2.medianBlur(mask_save, 5)
        #         cv2.imwrite('./../../pic/jigsaw_puzzle/demo/puzzle_cnt_proj_{}.png'.format(idx+1), mask_save)
        #         # cv2.namedWindow('debug', cv2.WINDOW_AUTOSIZE)
        #         # cv2.imshow('debug', mask)
        #         # cv2.waitKey(0)
        #
        # if len(pxls_template_proj) != 0:
        #     for idx, cnt in enumerate(pxls_template_proj):
        #         mask_save = np.zeros(mask_smoothed.shape[:2], dtype="uint8")
        #         cv2.polylines(mask_save, [cnt], True, (255, 255, 255), thickness=3)
        #         # cv2.drawContours(mask_save, [cnt], -1, (255, 255, 255), 3)
        #         mask_save = cv2.medianBlur(mask_save, 5)
        #         cv2.imwrite('./../../pic/jigsaw_puzzle/demo/template_cnt_proj_{}.png'.format(idx + 1), mask_save)

        # if len(template_cnts) != 0:
        #     cnt_mask = np.zeros(plate_mask.shape[:2], dtype="uint8")  # reset a blank image every time
        #     cv2.drawContours(cnt_mask, template_cnts, -1, (255, 255, 255), 3)
            # cv2.namedWindow('debug', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('debug', cnt_mask)
            # cv2.waitKey(1)

        # if len(template_cnts) == 4:
        #     cv2.imwrite('./color_image.png', color_image)
        #     cv2.imwrite('./mask.png', mask)
        #     np.save('./depth_raw_crop.npy', depth_image[pxl_min[1]: pxl_max[1], pxl_min[0]: pxl_max[0]])
        #     cv2.imwrite('./mask_table.png', mask_smoothed)
        #     cnt_mask = np.zeros(mask_smoothed.shape[:2], dtype="uint8")  # reset a blank image every time
        #     cv2.drawContours(cnt_mask, puzzle_cnts, -1, (255, 255, 255), 3)
        #     cv2.imwrite('./cnt_puzzle.png', cnt_mask)
        #     cnt_mask = np.zeros(plate_mask.shape[:2], dtype="uint8")  # reset a blank image every time
        #     cv2.drawContours(cnt_mask, template_cnts, -1, (255, 255, 255), 3)
        #     cv2.imwrite('./cnt_template.png', cnt_mask)

finally:
    # Stop streaming
    pipeline.stop()

# image = cv2.imread('./demo.png')
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# original = image.copy()
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(image, lower_hsv, upper_hsv)
# result = cv2.bitwise_and(original,original,mask=mask)
# # result[mask==0] = (255,255,255)
# #
# # # Make text black and foreground white
# # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# # result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
#
# cv2.imshow('mask', mask)
# cv2.imshow('result', result)
# cv2.waitKey()
