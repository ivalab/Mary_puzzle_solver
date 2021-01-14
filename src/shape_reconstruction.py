import numpy as np
import cv2
from PIL import Image

from shape_match_fd import getContours

def circle_recons(shape, contour):
    center = (int(np.mean(contour[:, :, 0])), int(np.mean(contour[:, :, 1])))

    radius = np.amax(np.sqrt(np.sum((contour - center) ** 2, axis=-1)))

    mask = np.zeros(shape, dtype="uint8")
    mask = cv2.circle(mask, center, int(np.ceil(radius)), 255, thickness = 2)

    return mask

def rect_recons(shape, contour):
    center = [np.mean(contour[:, :, 0]), np.mean(contour[:, :, 1])]

    dst = np.sqrt(np.sum((contour - center) ** 2, axis=-1))
    vertices = contour[dst.argsort()[-3:][::-1], :, :][:4, :, :]

    mask = np.zeros(shape, dtype="uint8")
    pts = np.array([[vertices[0, 0, 0], vertices[0, 0, 1]], \
                    [vertices[1, 0, 0], vertices[1, 0, 1]], \
                    [vertices[2, 0, 0], vertices[2, 0, 1]], \
                    [vertices[3, 0, 0], vertices[3, 0, 1]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv.polylines(mask, [pts], True, 255)

    return mask

def tri_recons(shape, contour):
    center = [np.mean(contour[:, :, 0]), np.mean(contour[:, :, 1])]

    dst = np.sqrt(np.sum((contour - center) ** 2, axis=-1))[:, 0]
    vertices = contour[dst.argsort()[-3:][::-1], :, :]

    mask = np.zeros(shape, dtype="uint8")
    pts = np.array([[vertices[0, 0, 0], vertices[0, 0, 1]], \
                    [vertices[1, 0, 0], vertices[1, 0, 1]], \
                    [vertices[2, 0, 0], vertices[2, 0, 1]], \
                    [vertices[3, 0, 0], vertices[3, 0, 1]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv.polylines(mask, [pts], True, 255)

    return mask

def reconstruct_convex_hull(shape, img):
    '''
    reconstruct the shape by computing the convex hull of the input contour
    :param
        shape: shape of output mask
        img: input mask
    :return:
        mask: mask of reconstructed shape
    '''
    cnt = getContours(img)
    mask = np.zeros(shape, dtype="uint8")

    ####################
    #  convex hull
    ####################
    hull = cv2.convexHull(cnt[0], False)[:, 0, :]
    mask = cv2.polylines(mask, [hull], isClosed=True, color=255, thickness=2, lineType=8, shift=0)

    return mask

def test():
    puzzles = []
    templates = []
    puzzles_pil = []
    templates_pil = []

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
    order = ['circle', 'triangle', 'rectangle', 'diamond']

    for idx, puzzle in enumerate(puzzles):
        cnt = getContours(puzzle)
        mask = np.zeros(puzzle.shape[:2], dtype="uint8")
        # if order[idx] == 'circle':
        #     mask = circle_recons(puzzle.shape[:2], cnt[0])
        # elif order[idx] == 'triangle':
        #     mask = tri_recons(puzzle.shape[:2], cnt[0])
        # elif order[idx] == 'rectangle':
        #     mask = rect_recons(puzzle.shape[:2], cnt[0])
        # elif order[idx] == 'diamond':
        #     mask = rect_recons(puzzle.shape[:2], cnt[0])

        ####################
        #  convex hull
        ####################
        hull = cv2.convexHull(cnt[0], False)[:, 0, :]
        mask = cv2.polylines(mask, [hull], isClosed=True, color=255, thickness=3, lineType=8, shift=0)
        # mask = cv2.drawContours(mask, hull, -1, 255, 1)
        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.dilate(mask, kernel, iterations=2)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.namedWindow('reconstructed mask', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('reconstructed mask', mask)
        cv2.waitKey(0)

    for idx, template in enumerate(templates):
        cnt = getContours(template)
        mask = np.zeros(template.shape[:2], dtype="uint8")
        # if order[idx] == 'circle':
        #     mask = circle_recons(puzzle.shape[:2], cnt[0])
        # elif order[idx] == 'triangle':
        #     mask = tri_recons(puzzle.shape[:2], cnt[0])
        # elif order[idx] == 'rectangle':
        #     mask = rect_recons(puzzle.shape[:2], cnt[0])
        # elif order[idx] == 'diamond':
        #     mask = rect_recons(puzzle.shape[:2], cnt[0])

        ####################
        #  convex hull
        ####################
        hull = cv2.convexHull(cnt[0], False)[:, 0, :]
        mask = cv2.polylines(mask, [hull], isClosed=True, color=255, thickness=3, lineType=8, shift=0)
        # mask = cv2.drawContours(mask, hull, -1, 255, 1)
        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.dilate(mask, kernel, iterations=2)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.namedWindow('reconstructed mask', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('reconstructed mask', mask)
        cv2.waitKey(0)

if __name__ == '__main__':
    test()