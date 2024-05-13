import cv2
import math
import numpy as np
from Constants import *


def filterFrameHSV(frame, puckLowerBoundary, puckUpperBoundary, robotLowerBoundary, robotUpperBoundary):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maskPuck = cv2.inRange(hsv, puckLowerBoundary, puckUpperBoundary)
    maskRobot = cv2.inRange(hsv, robotLowerBoundary, robotUpperBoundary)
    filteredFramePuck = cv2.bitwise_and(frame, frame, mask=maskPuck)
    filteredFrameRobot = cv2.bitwise_and(frame, frame, mask=maskRobot)
    filteredFrame = cv2.bitwise_or(filteredFramePuck, filteredFrameRobot)
    return filteredFrame


def detectPuck(filteredFrame, lowerBoundary, upperBoundary):
    hsv = cv2.cvtColor(filteredFrame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerBoundary, upperBoundary)
    mask_blur = cv2.medianBlur(mask, 19)
    contours, hierarchy = cv2.findContours(mask_blur, 1, 2)
    if not contours:
        return ((0, 0), 0)
    cnt = contours[0]
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    return (x, y), radius


def markInFrame(frame, x, y, radius, color):
    # Convert to int.
    center = (int(x), int(y))
    radius = int(radius)
    # Draw a circle around the puck in the unfiltered image.
    cv2.circle(frame, center, radius, color, 2)
    return frame


def markRobotRectangle(frame):
    # Only needs top left and bottom right corner.
    cv2.rectangle(frame, (0, 0, CAMERA_FRAME_HEIGHT, CAMERA_FRAME_ROBOT_MAX_Y), (0, 0, 255), 1)
    return frame
