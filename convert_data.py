import numpy as np
import sys
import matplotlib.pyplot as plt

# COLORS: 0:purple 1:pink 2 :dark green 3:yellow 4:red 5:blue 6:lightgreen

Y_MAX = 540*2 #Sets the size of the window, based on the resolution of the camera.
EPS = 5 * np.pi/180 # Set how close to 90 degrees points need to be to be recognized as correct
#(like epsilon)

def get_polar(origin, point):
    """
    Just converts xy coordinates to polar coordinates, given two (x,y) points,
    the data point to convert and the origin point we want to use for polar
    coordinates
    """
    xd = point[0]-origin[0]
    yd = point[1]-origin[1]
    dist = np.sqrt(xd**2 + yd**2) #distance from point to origin, radius

    alpha = np.arctan(yd / xd) #Calculate the angle with arctangent

    if xd>=0 and yd>0: #Annoying coordinate cases so the sign of the angle makes sense
        theta = alpha
    elif xd<0 and yd>0:
        theta = np.pi+alpha
    elif xd>=0 and yd<0:
        theta = 2*np.pi +alpha
    elif xd <0 and yd<0:
        theta = alpha+np.pi
    return np.array([dist, theta]) #Returns an (r,0) pair instead of (x,y)


def find_theta(color, locs, data, radius):
    """
    This function is meant to fill in gaps. Parameters:
    color - an index from 0 to 6, specifices which color to process
    locs - a list of indexes where the values are missing
    data - the actual array of data, for all colors
    radius - the known radius of the color from its origin.

    Using these, it tries to deduce the angle that is missing, either by looking
    at the angle of the other colors where possible or by looking at the nearest
    angles we do know for this color and taking the average. This is messy, could
    do better.

    Also its broken lol
    """
    sizes = [] #This is gonna be a list of numbers, where each is the number of consecutive
    #missed points. So, for example, if locs is [4, 6,7,8, 10, 12,13,14,15,16], then
    #sizes should end up being [1, 3, 1, 5]

    if len(locs) > 0: #Don't bother if there are none
        #print("Color: ", color)
        i = 0
        while i < len(locs):
            loc = locs[i] #So loc is the index of the first missing point in the array

            #For each gap, first find the size of that gap
            size = 1 #Length of missing gap
            if i < len(locs)-1: #Stop if we include all the points
                while loc+size == locs[i+size]: #Checks for consecutive missing digits
                    size +=1
                    #print(size)
                    if i+size == len(locs): #Makes sure we don't count past the end of the array
                        break
            #print("Size: ", size)
            sizes.append(size)
            i += size


        for i in range(len(sizes)): #Now we are gonna loop through each group of missing points.
            size = sizes[i] #Size of the current group
            prev_numbers = sum(sizes[:i]) #total combined length of previous groups of missed points
            start = locs[prev_numbers] #Index in data of the start of the current group
            #print("Start: ", start)

            #This is the bit where we fill in data. For now,
            #Checking right angle inners. Otherwise,
            #just drawing a line from one edge to the other and
            #filling in points. Could be much smarter, we will see if we need it.
            #Should also maybe check if points are false detections


            lower_bound = start-1 #Point with real data before the blank spot
            upper_bound = start+size #Point with real data after blank spot
            lower_theta = data[lower_bound,color,1] #1 to get angle, not radius
            upper_theta = data[upper_bound,color,1]
            #print(lower_theta, upper_theta)
            gap = upper_theta - lower_theta
            dy = gap/(size+1)
            for j in range(size):
                if color == 1 and data[start+j,2,1] != -100: #If we are looking for pink and we know dark green
                    data[start+j,color,1] = (data[start+j,2,1] - (np.pi/2))%(2*np.pi)
                    data[start+j,color,0] = radius
                    #Get the point by adding -90 degree to dark green
                elif color == 1 and data[start+j,3,1] != -100: #If we are looking for pink and we know yellow
                    data[start+j,color,1] = (data[start+j,3,1] + (np.pi))%(2*np.pi)
                    data[start+j,color,0] = radius
                    #Get the point by adding 180 to yellow
                elif color == 2 and data[start+j,1,1] != -100: #If we are looking for dark green and we know pink
                    data[start+j,color,1] = (data[start+j,1,1] + (np.pi/2))%(2*np.pi)
                    data[start+j,color,0] = radius
                    #Get the point by adding 90 degree to pink
                elif color == 2 and data[start+j,3,1] != -100: #If we are looking for dark green and we know yellow
                    data[start+j,color,1] = (data[start+j,3,1] - (np.pi/2))%(2*np.pi)
                    data[start+j,color,0] = radius
                    #Get the point by adding -90 to yellow
                elif color == 3 and data[start+j,1,1] != -100: #Looking for yellow, know pink
                    data[start+j,color,1] = (data[start+j,1,1] + (np.pi))%(2*np.pi)
                    data[start+j,color,0] = radius
                    #Get the point by adding 180 degree to pink
                elif color == 3 and data[start+j,2,1] != -100: # Looking for yellow, know dark green
                    data[start+j,color,1] = (data[start+j,2,1] + (np.pi/2))%(2*np.pi)
                    data[start+j,color,0] = radius
                    #Get the point by adding 90 to dark green
                else: #uh oh
                    data[start+j,color,1] = lower_theta + ((j+1)*dy) #Just draw a line from the start of the gap to the end
                    data[start+j,color,0] = radius

    return data


def get_stats(color, data):
    """
    For a given color (again, index from 0 to 6), look through the data and get the
    mean and standard deviation of each axis (works for xy and r0)
    """
    #return mean and standard deviation, both as length 2 lists (x and y or r and 0)
    #Ignores -100 points, but not outliers
    xsum = 0
    ysum = 0
    detected_points = 0
    for i in range(len(data)):
        tslice = data[i]
        if tslice[color][0] != -100:
            xsum += tslice[color][0]
            ysum += tslice[color][1]
            detected_points += 1

    xav = xsum/detected_points
    yav = ysum/detected_points
    xdv = 0
    ydv = 0
    for i in range(len(data)):
        tslice = data[i]
        if tslice[color][0] != -100:
            xdv += (tslice[color][0] - xav)**2
            ydv += (tslice[color][1] - yav)**2
    if detected_points == 0:
        raise ValueError("No Points successfully detected for color %s, data impossible to process."%color)
    xdv = np.sqrt(xdv/detected_points)
    ydv = np.sqrt(ydv/detected_points)
    return [xav, yav], [xdv, ydv]

def filtered_average(color, data, dev=False):
    """
    Uses the average function above to intelligently guess the actual average from
    reliable data. Basically, ignores points that were missed or are obviously wrong
    """
    #First, need to find the origin
    avs, dvs = get_stats(color, data)

    #So next, remove outliers, then average of normal points is our origin
    #Note: not always xy, can be r0
    xsum_clean = 0
    ysum_clean = 0
    detected_points = 0

    for i in range(len(data)):
        tslice = data[i]
        if tslice[color][0] != -100:
            if abs(tslice[color][0]-avs[0]) < 3*dvs[0] and abs(tslice[color][1]-avs[1]) < 3*dvs[1]:
                xsum_clean += tslice[color][0]
                ysum_clean += tslice[color][1]
                detected_points += 1
    if detected_points == 0:
        raise ValueError("No Points successfully detected for color %s, data impossible to process."%color)
    average = [xsum_clean/detected_points, ysum_clean/detected_points] #Take average of reasonable points
    if dev:
        return average, dvs[0]
    return average

def fix_glitches(data):
    """
    This is where we try to fix points which were detected falsely, harder than missed points
    Takes in all of the data, returns the same data but with glitches cleaned up.

    Or at least it tries.

    This should probably be rewritten, and just work based on the radius, Convert all the points that aren't
    -100 to polar, then use get_stats to find outliers with weird radii. Don't know why I tried to do it like
    this.
    """
    for i in range(len(data)):
        t_slice = data[i]
        if t_slice[1,0] != -100 and t_slice[2,0] != -100 and t_slice[3,0] != -100: #Only replace glitches, not missed points
            dif1 = (t_slice[2,1] - t_slice[1,1])%(np.pi*2) #Loop through every point and its neighbors,
            dif2 = (t_slice[3,1] - t_slice[2,1])%(np.pi*2) #and see if its differences make sense
            if abs(dif1-(np.pi/2)) > EPS: #If the difference seems too big, probably a false detection
            #This if statement just says points 1 and 2 are farther apart than expected

                #Of the three points we care about, some number of them are wrong (could be 1, 2, or all 3)
                if abs(dif2-(np.pi/2)) > EPS:
                    #Either point two is wrong, or multiple points are wrong,
                    #because the dif between 1 and 2 is bad and the dif between
                    #2 and 3 is bad
                    dif3 = (t_slice[3,1] - t_slice[1,1])%(np.pi*2) #To check, lets see how far off
                    #points 1 and 3 are

                    if abs(dif3-(np.pi)) > 2*EPS:
                        #If the difference between 1 and 3 is also bad, probably at least two are wrong,
                        #so we should try to figure things out based on radii

                        #Ok so it seems to me like using radii would be better for the whole thing,
                        #but also at this point in main the data isn't in polar so
                        #what was I doing

                        r1, dv1 = filtered_average(1, data, dev=True)[0] #BUG: These are not actually polar at this point.
                        r2, dv2 = filtered_average(2, data, dev=True)[0] #kinda doesnt matter because this also isn't being
                        r3, dv3 = filtered_average(3, data, dev=True)[0] #run at this point
                        r_check = [abs(r1-t_slice[1,0])<2*dv1, abs(r2-t_slice[2,0])<2*dv2, abs(r3-t_slice[3,0])<2*dv3]
                        #This basically says if the radius is within two standard deviations of the mean, its probably fine
                        #well, its trying to say that. What it actually says is if the x coordinate is within two standard
                        #deviations of the mean thats probably fine, which is...nonsense.
                        #Also, if we could check this the whole time, why didn't we just like, use this?

                        #Anyway, then use these booleans to fix everything with math
                        if r_check == [True, False, False]:
                            #1 is correct
                            theta2 = (t_slice[1,1]+(np.pi/2))%(np.pi*2)
                            t_slice[2] = np.array([r2, theta2])
                            theta3 = (t_slice[1,1]+(np.pi))%(np.pi*2)
                            t_slice[3] = np.array([r3, theta3])
                        elif r_check == [False, True, False]:
                            #2 is correct
                            theta1 = (t_slice[2,1]-(np.pi/2))%(np.pi*2)
                            t_slice[1] = np.array([r1, theta1])
                            theta3 = (t_slice[2,1]+(np.pi/2))%(np.pi*2)
                            t_slice[3] = np.array([r3, theta3])
                        elif r_check == [False, False, True]:
                            #3 is correct
                            theta1 = (t_slice[3,1]-(np.pi))%(np.pi*2)
                            t_slice[1] = np.array([r1, theta1])
                            theta2 = (t_slice[3,1]-(np.pi/2))%(np.pi*2)
                            t_slice[2] = np.array([r2, theta2])
                        else:
                            #Everything is a disaster
                            t_slice[1] = np.array([-100, -100])
                            t_slice[2] = np.array([-100, -100])
                            t_slice[3] = np.array([-100, -100])


                    else:
                        #Just 2 is wrong
                        radius = filtered_average(2, data)[0]
                        #Angle is set to be the average of the prediction of the other two points
                        theta = 0.5 * (((t_slice[1,1]+(np.pi/2))%(np.pi*2)) + ((t_slice[3,1]-(np.pi/2))%(np.pi*2)))
                        t_slice[2] = np.array([radius, theta])
                else: #dif1 was wrong, dif2 was fine,
                    #So it must be 1 thats wrong
                    radius = filtered_average(1, data)[0]
                    #Angle is set to be the average of the prediction of the other two points
                    theta = 0.5 * (((t_slice[2,1]-(np.pi/2))%(np.pi*2)) + ((t_slice[3,1]-(np.pi))%(np.pi*2)))
                    t_slice[3] = np.array([radius, theta])
            elif abs(dif2-(np.pi/2)) > EPS:
                #3 is wrong, because 1 and 2 must be right
                radius = filtered_average(3, data)[0]
                #Angle is set to be the average of the prediction of the other two points
                theta = 0.5 * (((t_slice[1,1]+np.pi)%(np.pi*2)) + ((t_slice[2,1]+(np.pi/2))%(np.pi*2)))
                t_slice[3] = np.array([radius, theta])


    #This function is a disaster and should probably just be re-written. Sorry about that.
    return data

def main(align=False):
    """
    QUESTIONS:
    What was align and why is fix glitches where it is

    Organizes all of the functions above in order. Here's how it works:
    - First, we make sure there are files specified.
    - Next, uninvert the y axis because screens are terrible and y is always
    upside down.
    - Find reliable coordinates for purple using filtered average
    - (Possibly) correct points which were based on incorrect detections
        - I believe this is done first to avoid filling in missing points based on fake data
        - Also so that filling in missing points like, fills in false detections too
    - Go through the inner points (pink, gark green, and yellow):
        - If they were successfully detectedm convert that point to polar
        - If not, write down that this point was missed
        - Find the radius using the average radius, then find theta using the
        find_theta() function
    - Now we repeat the process with the orbiters
    - Finally, save the data.
    """
    if len(sys.argv) != 3:
        print("Enter target filename and output filename")
    else:
        data_raw = np.load(sys.argv[1])
        clean = data_raw.copy()
        print(clean.shape)


        #first things first, lets un-invert the y axis
        num_samples = len(clean)
        for t_slice in clean:
            for color in t_slice:
                if color[1] != -100:
                    color[1] = Y_MAX - color[1]

        #First, need to find the origin
        origin = filtered_average(0, clean)
        print(origin)

        # #Now we need to get the inner points
        #print(clean)

        if align:
            #"align" is set to false by default so it looks like I wasn't using this, probably because it was a disaster.
            #but the idea was this fixes points that were false detections.
            #I think I wanted to catch these first and label them as missed so that find_theta gets them later
            clean = fix_glitches(clean)


        missing_points = [[],[],[],[],[],[]] #Where we will store lists of missing points for each color
        for i in range(len(clean)):
            t_slice = clean[i] #Grab the xy coordinates of this slice in time
            p = t_slice[1]  #For pink
            dg = t_slice[2] #Dark green
            yl = t_slice[3] # Yellow
            #Those are the inner points, we have to fix them first because orbiters
            #depend on them



            if p[0] != -100: #If the point is detected
                p_coord = get_polar(origin, p) #convert to polar
                t_slice[1] = np.array(p_coord) #Store the new value
            else:
                missing_points[0].append(i) #Otherwise, note that this point is bad
            if dg[0] != -100:
                dg_coord = get_polar(origin, dg)
                t_slice[2] = np.array(dg_coord)
            else:
                missing_points[1].append(i)
            if yl[0] != -100:
                yl_coord = get_polar(origin, yl)
                t_slice[3] = np.array(yl_coord)
            else:
                missing_points[2].append(i)


        #print(missing_points)
        for color in range(3): #Still only looking at the first 3 colors, inner points
            radius = filtered_average(color, clean)[0]
            print(radius)
            clean = find_theta(color, missing_points[color], clean, radius) #Fill in missing points and put back in data
            #Before, this said "data" and not clean, could have been a problem? Maybe I just fixed everything
            #Or broke everything worse
            #wait does python do editing in place? Did any of this matter

        #Finally, orbiters.
        for i in range(len(clean)):
            t_slice = clean[i]
            p = t_slice[1] #Still need to reference locations of inner points for origins
            dg = t_slice[2]
            yl = t_slice[3]
            r = t_slice[4] #Grab the xy coordinates
            b = t_slice[5]
            lg = t_slice[6]

            if r[0] != -100: #If the point is detected
                r_coord = get_polar(p, r) #convert to polar (using related inner point as origin)
                t_slice[4] = np.array(r_coord) #Store the new value
            else:
                missing_points[3].append(i) #Otherwise, note that this point is bad
            if b[0] != -100:
                b_coord = get_polar(dg, b)
                t_slice[5] = np.array(b_coord)
            else:
                missing_points[4].append(i)
            if lg[0] != -100:
                lg_coord = get_polar(yl, lg)
                t_slice[6] = np.array(lg_coord)
            else:
                missing_points[5].append(i)

        print(missing_points)
        for color in range(3,6):
            radius = filtered_average(color, clean)[0]
            print(radius)
            data = find_theta(color, missing_points[color], clean, radius) #Fill in missing points and put back in data

        np.save(sys.argv[2], clean)
        #For debug visualization
        plt.plot(clean[:,1,1])
        plt.figure()
        plt.plot(clean[:,1,0])
        plt.show()

if __name__ == '__main__':
    main()
