# Chaotic Pendulum Color Tracking
This code runs color tracking algorithms on video files, then processes the (x,y)
data to both fill in missing points and convert to polar coordinates.


## Files
A description of all of the files found in this repository
### Most important:

- videotrack.py - the most up-to-date version of the color tracking code. Opens
an mp4, tracks all seven colors, and logs those points into one of the data.npy.
To run, enter "python videotrack.py" in the terminal, or "python videotrack.py -c"
for calibration mode, which allows you to adjust the colors of each sticker based
on a stillframe from one of the recordings. Note, calibration mode is a little
bit broken right now. Working on that.

- convert_data.py - reads the raw data from videotrack.py and converts it to the
form we want, filling in gaps, and stores the results in cleaned_data.npy. This
is currently the main focus of the project, getting this code to actually work.

- display_data.py - takes the cleaned data file and shows it on a graph. Right
now, this graph is a regular xy kind of thing, even though the coordinates are
polar. Really just for debugging at this point.

- colors.csv - this is where color information is stored, and what is edited when
you run calibration mode. Color data uses HSV format inside videotrack. The
numbers stored in colors.csv are the upper and lower bounds of the Hue value.
(Messing with these values seems like a good candidate for cleaning up the data)

### Other files:
- tracking.py - great place to learn how this thing works. This was where I got
color tracking to work, so it's relatively simple (still not simple).This tracks
one color which you calibrate at the start and just graphs its location. It
should have a lot of answers, and might be useful for software setup.

- livetrack.py - basically the same as videotrack.py, except it loads in webcam
footage instead of a pre-recorded file. Probably not useful, I was using this to
figure out saving the data.

- A bunch of .mp4s and .cines (and .m4vs I guess)- These are the video files that
contain the data. There are three fast one (1.5, 2, and 3, obviously) and 2 slow
ones. Ideally we eventually get a cleaned_data file for each of these that makes
sense and doesn't have gaps. This will be...challenging.

- frame1.png - A single frame from one of the videos. Mostly useful for checking
where the stickers are relative to each other.

## Data Structure and Terms
At each frame, the color tracking software writes down the x and y locations of
the detected color, and then stores each pair in a list. This means that each frame
results in a list of 7 (x,y) pairs. The order used in this list (and everywhere
else) is [purple, pink, dark green, yellow, red, blue, light green]. Then, this
list is appended to the actual data matrix, resulting in three dimensional data.
Accessing a specific point in this data looks like data[frame, color, x/y]. Often,
the code will grab one frame of data (usually called something like t_slice, like
one slice of time) by just using the first index, data[0] should give the xy
coordinates of each color during the first frame. To get the coordinates for one
color, similarly, two indexes are used (for example, data[0,0] gets purple's position
during the first frame). If the color tracking algorithm fails to detect a point,
it writes -100 for both the x and y values. This makes detecting missing points
very easy.

The output format for the data is very similar, the only change is that x and y
are replaced with r and theta, otherwise they are organized exactly the same way.
There is one complication: For pink, dark green, and yellow, the radius and angle
are calculated with the purple sticker defined as the origin. This makes sense as
this is the point they rotate around. For red, dark green, and light green, however,
they do not rotate (directly) around the center, but around pink, dark green, and
yellow respectively. This means that their polar coordinates are calculated with
their own center of rotation as the origin. This means that for all of the points,
the radius should be basically constant. This is a useful way to find false
detections.

### Calibration
In order to reset the values of a given color, each color tracking script is
equipped with a "calibration mode," where it accepts an (x,y) coordinate and color
name from the terminal, which it uses to extract color information from that location
in the video. A good method is to mouseover the point in on the video playback
where the color you want to detect is, then read off the coordinates in the lower
left hand corner. The input format is color name,x,y. To see a list of all color
names, look at colors.csv, which is the file updated by the calibration script.

### Important Terms
(like, that I made up and don't just mean something google-able)
- Inner Points - One of the three points connected to the central purple hub at
fixed angles
- Orbiters/outer points - one of the three points connected to the inner points
which rotate freely   
- Missed point - A data point which the color tracking algorithm failed to detect.
It writes (-100, -100) in place of actual (x,y) coordinates.
- False detections/glitch - A data point which the color tracking algorithm
detected, but incorrectly, usually when the colored sticker is briefly offscreen.
These are harder to detect, as they still look like data, but after converting to
polar coordinates the radius should be very different from most points.

## To Do:
- Re-write fix glitches, it is currently nonsense. This might be a good starting
point once things are set up and running.
- Get reasonable data. This means:
  - No missed points or false detections after convert_data, and points that are filled
  in make sense.
  - As few filled in points as possible, this may involve improving/tuning the color
  tracking algorithm, possible by changing the windows around the color. Might be useful
  to somehow visualize the HSV values of a screenshot to see what the algorithm is
  seeing.
  - A cleaned data file for each of the four videos (fast2, fast3, slow1, and slow2).
    - I will probably rename the fast ones this is silly
