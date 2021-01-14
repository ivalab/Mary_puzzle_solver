import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco

# default transformation matrix from camera to aruco tag
default_M_CL = np.array([[8.05203923e-04,  -9.98258274e-01,  5.89895796e-02, 2.11182116e-02],
                         [-7.25197650e-01, -4.11996435e-02, -6.87307033e-01, 1.19383476e-01],
                         [6.88540282e-01,  -4.22256822e-02, -7.23967728e-01, 5.68361874e-01],
                         [0.00000000e+00,   0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

# Realsense D435 depth camera intrinsic matrix
cameraMatrix = np.array([[613.8052368164062, 0.0, 328.09918212890625],
                         [0.0, 613.8528442382812, 242.4539337158203],
                         [0.0, 0.0, 1.0]])
distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# transformation matrix from base to aruco tag
M_BL = np.array([[1., 0., 0.,  0.28945],
                 [0., 1., 0.,  0.29145],
                 [0., 0., 1.,  0.00000],
                 [0., 0., 0.,  1.00000]])

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

    print(M_CL)

    if visualize:
        # print('aruco is located at mean position (%d, %d)' %(mean_x ,mean_y))
        aruco.drawAxis(image_init, cameraMatrix, distCoeffs, rvec_CL, tvec_CL, markerLength_CL)
    return M_CL

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    # fgbg = cv2.createBackgroundSubtractorMOG2()

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
        depth_raw = np.array(depth_frame.get_data()) * depth_scale
        color_image = np.array(color_frame.get_data())
        gray = color_image.astype(np.uint8)

        depth_frame = (depth_raw / depth_scale).astype(np.uint8)


        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))

        M_CL = get_M_CL(gray, color_image, True)

        # test if aruco tag detection is accurate
        u = 495
        v = 272
        depth = depth_raw[v, u]
        q_C = np.linalg.inv(cameraMatrix).dot(np.array([u*depth, v*depth, depth]))
        q_L = np.linalg.inv(M_CL).dot(np.array([q_C[0], q_C[1], q_C[2], 1]))
        q_B = M_BL.dot(q_L)
        print("depth: ", depth)
        print("coordinate of center of aruco tag relative to aruco tag: ", q_L)
        print("coordinate of center of aruco tag relative to base: ", q_B)

        # Show images
        cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('depth', depth_frame)
        cv2.namedWindow('rgb', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('rgb', color_image)
        # cv2.imwrite('./demo.png', color_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()