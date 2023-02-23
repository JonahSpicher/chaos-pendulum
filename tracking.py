import numpy as np
import time
import imutils
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import psutil
import collections


## TODO:
"""
Things to fix:
- Write data smoothing (using known fixed distances and stuff)

"""


class Model:
    """
    Class that keeps track of what is going on in the program. It knows the current color
    of the drawing tool, and things like whether the program should be quitting. Also,
    the init of the model is where the elements of the interface are created (not displayed)
    """
    def __init__(self):
        self.frame = None
        self.upper_color = np.zeros(1)
        self.lower_color = np.zeros(1)
        self.calibration_start = 0
        self.elapsed_time = 0
        self.calibration_time = 10
        self.current_path = os.path.dirname(__file__)
        self.line_points = []
        self.cursor = ()
        self.state = 'calibration'
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.calib_count = 0

        self.xdata = np.zeros(100)
        self.ydata = np.zeros(100)

    def show_data(self, x, y):
        if np.sum(self.xdata) == 0: #Just for display, makes the graph look nice on opening
            for i in range(len(self.xdata)):
                self.xdata[i] = x
                self.ydata[i] = 480-y
        else:
            self.xdata = np.delete(self.xdata, 0)
            self.xdata = np.append(self.xdata, x)
            self.ydata = np.delete(self.ydata, 0)
            self.ydata = np.append(self.ydata, 480-y)

        self.ax.clear()
        self.ax.set_xlim(0, 630)
        self.ax.set_ylim(0, 480)
        self.ax.plot(self.xdata, self.ydata)
        plt.draw()
        #plt.pause(0.001)


class Controller:
    """
    A class to detect input from the user of the program. It takes the model as input, and has a
    function to return the position of an object of the color from the calibration step
    """
    def __init__(self, model):
        self.model = model

    def check_distance(self, point):
        """
        Checks if a point is within a certain distance of the last point in line_points.
        This gets rid of false positives far away from where the cursor was last frame.
        """
        if self.model.line_points:
            if self.model.line_points[-1]:
                x1 = point[0]
                y1 = point[1]
                x2 = self.model.line_points[-1][0]
                y2 = self.model.line_points[-1][1]
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                return distance
        else:
            return 0

    def detect_wand(self, lower, upper):
        """
        Looks at the current frame, finds the largest contour of the target color,
        and returns the center of that point, as well as the current color and
        velocity of the cursor.
        """
        kernel = np.ones((15, 15), 'uint8') # make a kernel for blurring
        blurred = cv2.GaussianBlur(self.model.frame, (17, 17), 0)
        blurred = cv2.dilate(self.model.frame, kernel) # blur the frame to average out the value in the circle
        hsv_frame = cv2.cvtColor(self.model.frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower, upper)
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        center = None

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                center = (0,0)

            if radius > 30:
                velocity = self.check_distance(center)
                return((center[0], center[1])) #Tuple where line info is stored)
            else:
                return False

class View:
    """
    Class that creates and then draws on the display. Here is where you draw draw_lines
    and tell the program to display interface elements. If we add future tools besides the line,
    it will go here.
    """
    def __init__(self, model):
        self.model = model


    def show_position(self):
        """
        Iterates through all the points in the list of where the target has been
        and draws rectangles between them.
        """
        if self.model.cursor is not None:
            if self.model.cursor is not False:
                cv2.putText(self.model.frame,'Position: ('+ str(self.model.cursor[0]) + ',' + str(self.model.cursor[1]) + ')',(30,30),cv2.FONT_HERSHEY_DUPLEX,1,(255, 255, 255))

            else:
                cv2.putText(self.model.frame, 'Position: Not Found', (30,30),cv2.FONT_HERSHEY_DUPLEX,1,(255, 255, 255))


    def show_cursor(self):
        """
        Shows a circle on the screen where the cursor is. The circle matches the current thickness
        and color of the cursor. If the current tool is erase, the circle is grey instead.
        """

        if self.model.cursor: #drawing cursor
            cv2.circle(self.model.frame, ((self.model.cursor[0]),(self.model.cursor[1])),5,(0,0,0), thickness = 2)


def process_frame(model, controller, view):
    """
    This function finds the cursors, executes what current tool needs to happen,
    shows all the buttons, and draws on the frame.
    """
    model.frame = cv2.flip(model.frame,1) # reverse the frame so people aren't confused
    if model.state == 'tracking':
        model.cursor = controller.detect_wand(model.lower_color, model.upper_color) # find both cursors
        view.show_position()
        view.show_cursor()
        if model.cursor is not None and model.cursor is not False:
            #print(model.cursor[0], model.cursor[1])
            model.show_data(model.cursor[0], model.cursor[1])
        model.line_points.append(model.cursor) # if the program is drawing, it simply needs to add to the list of points to be drawn.


    if model.state == 'calibration': # all this  or model.tool =='calibration 2'code only needs to run if the program is currently calibrating
        model.elapsed_time = time.time() - model.calibration_start
        if model.elapsed_time < model.calibration_time:
            cv2.putText(model.frame,'Place calibration color in center:' + str(int(model.calibration_time - model.elapsed_time)),(30,30),cv2.FONT_HERSHEY_DUPLEX,1,(255, 255, 255))
            cv2.circle(model.frame, (int(model.frame.shape[1]/2), int(model.frame.shape[0]/2)), 50,(255,255,255), thickness = 3)
            cv2.circle(model.frame, (int(model.frame.shape[1]/2), int(model.frame.shape[0]/2)), 55,(0,0,0), thickness = 3)
        elif model.elapsed_time > model.calibration_time:
            kernel = np.ones((15, 15), 'uint8') # make a kernel for blurring
            model.frame = cv2.dilate(model.frame, kernel) # blur the frame to average out the value in the circle
            model.frame = cv2.GaussianBlur(model.frame, (17, 17), 0)
            hsv_frame = cv2.cvtColor(model.frame, cv2.COLOR_BGR2HSV)
            pixel = hsv_frame[int(model.frame.shape[0]/2), int(model.frame.shape[1]/2)] # grab the pixel from the center of the calibration circle

            model.lower_color, model.upper_color = (np.array([pixel[0]-10,50,50]), np.array([pixel[0]+10,250,250]))
            model.elapsed_time = 0
            model.calibration_start = time.time()
            model.calib_count += 1
            # model.state = 'tracking'
            print(model.lower_color, model.upper_color)
            model.state = 'tracking'




def main_loop():
    """
    If this script is run directly, this loop takes care of making a window for the
    program and initializing everything. If the program is being run as a web app,
    this function is not used.
    """
    model = Model()
    view = View(model)
    controller = Controller(model)
    cap = cv2.VideoCapture(0)
    model.calibration_start = time.time()
    #while cv2.getWindowProperty(cap, 0) >=0:
    plt.ion()
    plt.show()
    while True:
        #keyCode = cv2.waitKey(50)
        _, model.frame = cap.read() # get a frame from the camera
        process_frame(model,controller,view) # this is where all the work is done.
        cv2.imshow('pendulum tracker',model.frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or model.state == 'exit':
            cap.release()
            cv2.destroyAllWindows()
            break

    # data = list(filter(None, model.line_points))
    # print(data)
    # xdata = [pair[0] for pair in data]
    # ydata = [480-pair[1] for pair in data]
    # plt.plot(xdata, ydata)
    # plt.show()


if __name__ == '__main__':
    # again, this doesn't run unless this script is run directly, not from the browser
    main_loop()
