import numpy as np
import time
import imutils
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import psutil
import collections
import sys
import threading
import queue
import csv


## TODO:
"""
Things to fix:
- speed up framerate - Could run on saved footage from a high speed camera
- Write data smoothing (using known fixed distances and stuff)

"""
def add_input(input_queue):
    """
    Process text input without blocking code during calibration
    """
    while True:
        input_queue.put(sys.stdin.read(1))



def set_color(cname, lval, hval):
    """
    Updates the .csv file with color values, given a color to update and the value range
    """
    contents = []
    with open('colors.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            contents.append(row)
    for row in contents:
        if row[0] == cname:
            row[1] = lval
            row[2] = hval
            break

    with open('colors.csv', mode='w') as file:
        cwrite = csv.writer(file)
        cwrite.writerows(contents)
    print("Updated "+cname+" with values %d and %d" %(lval, hval))

class Model:
    """
    Class that keeps track of what is going on in the program. It knows the current color
    of the drawing tool, and things like whether the program should be quitting. Also,
    the init of the model is where the elements of the interface are created (not displayed)
    """
    def __init__(self):
        self.frame = None

        #Initialize where we store the colors
        self.sbounds = (50,250)
        self.vbounds = (75,250)
        self.red_lower = np.zeros(3)
        self.red_upper = np.zeros(3)
        self.yellow_lower = np.zeros(3)
        self.yellow_upper = np.zeros(3)
        self.blue_lower = np.zeros(3)
        self.blue_upper = np.zeros(3)
        self.dgreen_lower = np.zeros(3)
        self.dgreen_upper = np.zeros(3)
        self.purple_lower = np.zeros(3)
        self.purple_upper = np.zeros(3)
        self.lpink_lower = np.zeros(3)
        self.lpink_upper = np.zeros(3)
        self.lgreen_lower = np.zeros(3)
        self.lgreen_upper = np.zeros(3)
        self.upper_bounds = [self.red_upper, self.yellow_upper, self.blue_upper,
                            self.dgreen_upper,
                            self.purple_upper, self.lpink_upper,
                            self.lgreen_upper]
        self.lower_bounds = [self.red_lower, self.yellow_lower, self.blue_lower,
                            self.dgreen_lower,
                            self.purple_lower, self.lpink_lower,
                            self.lgreen_lower]

        #Now fill them in from the csv
        self.load_colors()



        self.current_path = os.path.dirname(__file__)
        self.line_points = [None]*len(self.upper_bounds)
        self.cursors = [None]*len(self.upper_bounds)
        self.display_color = 0
        self.state = 'tracking'
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.plot_colors = ['r', 'y', 'b', 'g', 'k', 'm', 'c']

        self.color_start = 0
        self.elapsed_time = 0

        self.xdata = np.zeros((len(self.upper_bounds), 100))
        self.ydata = np.zeros((len(self.upper_bounds), 100))
        self.all_data = np.zeros((1, len(self.lower_bounds), 2))
        self.t = 0 #Tracks the time dimension

    def load_colors(self):
        """
        Reads the csv file to update color calibration
        """
        with open('colors.csv', mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                self.lower_bounds[line_count] = np.array([int(row[1]), self.sbounds[0], self.vbounds[0]])
                self.upper_bounds[line_count] = np.array([int(row[2]), self.sbounds[1], self.vbounds[1]])
                line_count += 1
        #print(self.lower_bounds, self.upper_bounds)

    def process_input(self, input_buffer):
        """
        Gets the color value at a specified location in the image
        """
        command = ''.join(input_buffer)[:-1].replace('\n','').replace(' ','').split(',')

        kernel = np.ones((15, 15), 'uint8') # make a kernel for blurring
        self.frame = cv2.dilate(self.frame, kernel) # blur the frame to average out the value in the circle
        self.frame = cv2.GaussianBlur(self.frame, (17, 17), 0)
        hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        pixel = hsv_frame[int(command[1]), int(command[2])] # grab the pixel from the center of the calibration circle

        l, h = (pixel[0]-10, pixel[0]+10)
        command[1] = l
        command[2] = h

        return command

    def show_data(self, x, y, color):
        """
        displays all tracked points as lines on a graph. Kind of.
        Probably only works on live capture, bet its sketchy for pre-recorded video
        """
        #print(color)
        if np.sum(self.xdata) == 0: #Just for display, makes the graph look nice on opening
            for i in range(len(self.xdata)):
                self.xdata[color][i] = x
                self.ydata[color][i] = 480-y
        else:
            x_copy = self.xdata[color]
            y_copy = self.ydata[color]
            x_copy = np.delete(self.xdata[color], 0)
            x_copy = np.append(x_copy, x)
            y_copy = np.delete(self.ydata[color], 0)
            y_copy = np.append(y_copy, 480-y)
            self.xdata[color] = x_copy
            self.ydata[color] = y_copy

        if color == 0:
            self.ax.clear()
            self.ax.set_xlim(0, 630)
            self.ax.set_ylim(0, 480)

        self.ax.plot(self.xdata[color], self.ydata[color], color=self.plot_colors[color])
        plt.draw()



        #plt.pause(0.001)


class Controller:
    """
    A class to detect input from the user of the program. It takes the model as input, and has a
    function to return the position of an object of the color from the calibration step
    """
    def __init__(self, model):
        self.model = model

    def check_distance(self, point, color):
        """
        Checks if a point is within a certain distance of the last point in line_points.
        This gets rid of false positives far away from where the cursor was last frame.
        """
        if self.model.line_points[color]:
            if self.model.line_points[color][-1]:
                x1 = point[0]
                y1 = point[1]
                x2 = self.model.line_points[color][-1][0]
                y2 = self.model.line_points[color][-1][1]
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                return distance
        else:
            return 0

    def detect_wand(self, lower, upper, color):
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
                velocity = self.check_distance(center, color)
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
        if self.model.cursors[self.model.display_color] is not None:
            if self.model.cursors[self.model.display_color] is not False:
                cv2.putText(self.model.frame,'Position: ('+ str(self.model.cursors[self.model.display_color][0]) + ',' + str(self.model.cursors[self.model.display_color][1]) + ')',(30,30),cv2.FONT_HERSHEY_DUPLEX,1,(255, 255, 255))

            else:
                cv2.putText(self.model.frame, 'Position: Not Found', (30,30),cv2.FONT_HERSHEY_DUPLEX,1,(255, 255, 255))


    def show_cursor(self):
        """
        Shows a circle on the screen where the cursor is. The circle matches the current thickness
        and color of the cursor. If the current tool is erase, the circle is grey instead.
        """

        if self.model.cursors[self.model.display_color]: #drawing cursor
            cv2.circle(self.model.frame, ((self.model.cursors[self.model.display_color][0]),(self.model.cursors[self.model.display_color][1])),5,(0,0,0), thickness = 2)


def process_frame(model, controller, view):
    """
    This function finds the cursors, executes what current tool needs to happen,
    shows all the buttons, and draws on the frame.
    """
    model.frame = cv2.flip(model.frame,1) # reverse the frame so people aren't confused
    if model.state == 'tracking':
        for color in range(len(model.upper_bounds)):
            model.cursors[color] = controller.detect_wand(model.lower_bounds[color], model.upper_bounds[color], color) # get red point
        # view.show_position()

            if model.cursors[color] is not None and model.cursors[color] is not False:
                #print(model.cursor[0], model.cursor[1])
                x = model.cursors[color][0]
                y = model.cursors[color][1]
                model.all_data[model.t,color,0] = x
                model.all_data[model.t,color,1] = y


                model.show_data(x, y, color)
            else:
                model.all_data[model.t,color,0] = -100
                model.all_data[model.t,color,1] = -100


            if model.line_points[color] is None:
                model.line_points[color] = [0]
            model.line_points[color].append(model.cursors[color]) # if the program is drawing, it simply needs to add to the list of points to be drawn.
        #view.show_cursor()
        model.all_data = np.append(model.all_data,np.zeros((1,len(model.lower_bounds), 2)),axis=0)
        model.t+=1






def main_loop(calibrate=False, fname='data.npy'):
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
    model.color_start = time.time()
    current_color = 0
    input_buffer = []

    if calibrate:
        input_queue = queue.Queue()

        input_thread = threading.Thread(target=add_input, args=(input_queue,))
        input_thread.daemon = True
        input_thread.start()
        last_update = time.time()

    while True:
        #keyCode = cv2.waitKey(50)
        if calibrate:
            if time.time()-last_update>0.5:
                sys.stdout.write(".")
                last_update = time.time()
            if not input_queue.empty():
                #print("\ninput:", input_queue.get())
                input_buffer.append(input_queue.get())
            else:
                if len(input_buffer) > 0:
                    #print(input_buffer) #Not fast, but this always contains the last typed command

                    command = model.process_input(input_buffer)

                    set_color(command[0], command[1], command[2])
                    model.load_colors() #re access the .csv file
                    input_buffer = [] #reset the input buffer so we stop doing this
                    print("Enter <color, y_coord, x_coord> to calibrate new color.")

        _, model.frame = cap.read() # get a frame from the camera
        process_frame(model,controller,view) # this is where all the work is done.
        cv2.imshow('pendulum tracker',model.frame)
        model.elapsed_time = time.time() - model.color_start #Just to be moved to a useful place
        if model.elapsed_time >= 15:
            #model.display_color+=1 #Cycle display colors
            model.color_start = time.time()
            if model.display_color == len(model.upper_bounds):
                model.display_color = 0
        if cv2.waitKey(1) & 0xFF == ord('q') or model.state == 'exit':
            np.save(fname, model.all_data)
            cap.release()
            cv2.destroyAllWindows()
            break



if __name__ == '__main__':
    # again, this doesn't run unless this script is run directly, not from the browser
    if len(sys.argv) >1:
        if str(sys.argv[1]) == '-c':
            print("Calibration Mode.\nEnter <color, y_coord, x_coord> to calibrate new color.")
            main_loop(calibrate=True,fname='data.npy')
        else:
            print("Invalid argument")
    else:
        main_loop(fname='data.npy')
