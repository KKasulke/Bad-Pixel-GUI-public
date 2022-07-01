# import required libraries
import numpy as np
import cv2

def maskGeneration(IMAGE_SIZE_X, IMAGE_SIZE_Y, masks_path, num_masks):
    # define mask parameters
    MIN_NUMBER_LINES = 5
    MAX_NUMBER_LINES = 12
    MAX_LINE_THICKNESS = 4
    MIN_NUMBER_CIRCLES = 4
    MAX_NUMBER_CIRCLES = 10
    MIN_CIRCLE_RADIAN = 4
    MAX_CIRCLE_RADIAN = 10

    for maskIndex in range(0, num_masks):
        # create empty mask
        mask = 255*np.ones([IMAGE_SIZE_Y, IMAGE_SIZE_X], np.uint8)

        # create random lines and rectangles
        numLines = np.random.randint(MIN_NUMBER_LINES, MAX_NUMBER_LINES, 1)[0]
        numCicles = np.random.randint(MIN_NUMBER_CIRCLES, MAX_NUMBER_CIRCLES, 1)[0]

        # create lines on mask
        linePointsX = np.random.randint(0,IMAGE_SIZE_X, [numLines, 2])
        linePointsY = np.random.randint(0,IMAGE_SIZE_Y, [numLines, 2])
        lineThicknesses = np.random.randint(1, MAX_LINE_THICKNESS, numLines)
        for line in range(numLines):
            cv2.line(mask, (linePointsX[line,0], linePointsY[line,0]),
                (linePointsX[line,1], linePointsY[line,1]), 0, 
                lineThicknesses[line])

        # create circles on mask
        centersX = np.random.randint(0,IMAGE_SIZE_X, numCicles)
        centersY = np.random.randint(0,IMAGE_SIZE_Y, numCicles)
        radii = np.random.randint(MIN_CIRCLE_RADIAN, MAX_CIRCLE_RADIAN, numCicles)

        # draw filled circles
        for circle in range(numCicles):
            cv2.circle(mask, (centersX[circle],centersY[circle]),
                radii[circle], 0, -1)

        # save current mask
        cv2.imwrite(masks_path + '\\mask-' + f'{maskIndex:06}' + '.png', mask)
    return