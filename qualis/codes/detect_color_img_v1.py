import numpy as np
import imutils
import cv2


# define the lower and upper boundaries of the colors in the HSV color space
lower = {'aço': (159,141,118),
         'bar': (72,72,70)}

upper = {'aço': (113,106,102),
         'bar': (42,44,43)}

#NOIR
lower = {'aço': (23,9,31),
         'bar': (14,6,21)}

upper = {'aço': (39,14,44),
         'bar': (52,20,50)}

# define standard colors for circle around the object
colors = {'aço': (0, 0, 255),
          'bar': (255, 0, 0)}

camera = cv2.VideoCapture(0)

while True:
    # read the video in real time
    _, frame = camera.read()

    # resize the frame
    frame = imutils.resize(frame, width=600)
    # transform in to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # for each color in dictionary check object in frame
    for key, value in upper.items():
        # construct a mask for the color from dictionary`1, then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size. Correct this value for your obect's size
            if radius > 0.5:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)),
                           int(radius), colors[key], 2)
                cv2.putText(frame, key + " object", (int(x-radius), int(y - radius)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[key], 2)

    # show the frame to our screen
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()