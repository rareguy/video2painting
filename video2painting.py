import cv2
import glob
import os
import time
import argparse
import math
import progressbar
import sys
from pointillism import *
from numba import jit
import makeasciiart
import traceback

###
# cleaning up warnings that littered the output
import warnings
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

###
# Functions that are useful for readability... maybe
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

list_of_mode = ['oil', 'point']

def toOilPainting(img):
	return cv2.xphoto.oilPainting(img, 8, 1)
	
def toPointillismPainting(img):
	stroke_scale = int(math.ceil(max(img.shape) / 250))
	#print("Automatically chosen stroke scale: %d" % stroke_scale)
	
	gradient_smoothing_radius = int(round(max(img.shape) / 50))
	#print("Automatically chosen gradient smoothing radius: %d" % gradient_smoothing_radius)

	# convert the image to grayscale to compute the gradient
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#print("Computing color palette...")
	palette = ColorPalette.from_image(img, 20)

	#print("Extending color palette...")
	palette = palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])

	#print("Computing gradient...")
	gradient = VectorField.from_gradient(gray)

	#print("Smoothing gradient...")
	gradient.smooth(gradient_smoothing_radius)

	#print("Drawing image...")
	# create a "cartonized" version of the image to use as a base for the painting
	res = cv2.medianBlur(img, 11)
	# define a randomized grid of locations for the brush strokes
	grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
	batch_size = 10000

	#bar = progressbar.ProgressBar()
	for h in range(0, len(grid), batch_size):
		# get the pixel colors at each point of the grid
		pixels = np.array([img[x[0], x[1]] for x in grid[h:min(h + batch_size, len(grid))]])
		# precompute the probabilities for each color in the palette
		# lower values of k means more randomnes
		color_probabilities = compute_color_probabilities(pixels, palette, k=9)

		for i, (y, x) in enumerate(grid[h:min(h + batch_size, len(grid))]):
			color = color_select(color_probabilities[i], palette)
			angle = math.degrees(gradient.direction(y, x)) + 90
			length = int(round(stroke_scale + stroke_scale * math.sqrt(gradient.magnitude(y, x))))

			# draw the brush stroke
			cv2.ellipse(res, (x, y), (length, stroke_scale), angle, 0, 360, color, -1, cv2.LINE_AA)

	return res

def toASCIIArt(filename):
	# not yet implemented, the library is buggy
	origin, newimg = makeasciiart.img2ascii(filename)
	return newimg

parser = argparse.ArgumentParser(description='this script will convert your normal video to painting-style video. example usage: video2painting.py oil video.mp4')
parser.add_argument('mode', metavar='mode', type=str, help='the video conversion mode ('+ ", ".join(list_of_mode)+ ")")
parser.add_argument('source_video', metavar='source', type=str, help='the video filename (currently only mp4) put the source ONLY IN CURRENT DIRECTORY, if you use exact path it will bug out')
parser.add_argument('--fps', metavar='fps', type=int, nargs="?", help='custom fps to make the visual looks cooler', default=0)
args = parser.parse_args()

if not (args.mode in list_of_mode):
	print(args.mode, "is not available")
	assert False


inputfile = args.source_video
print("processing", inputfile, "...")	
try:
	# img = cv2.imread('img.jpg')
	# res = cv2.xphoto.oilPainting(img, 7, 1)
	target_fps = 0
	# Opens the Video file
	cap= cv2.VideoCapture(inputfile)
	filename = inputfile[:-4]
	fps = cap.get(cv2.CAP_PROP_FPS)
	if args.fps > 0 and args.fps <= fps:
		target_fps = args.fps
	else:
		target_fps = fps
	print("Original FPS:", fps)
	print("Target FPS:", target_fps)
	
	i = 0
	print("converting video to frames...")
	try:
		os.mkdir("temp"+str(filename))
	except:
		print("i think there is already a folder named temp, pls delete it")
		sys.exit(1)
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == False:
			break
		cv2.imwrite('temp'+str(filename)+'/temp'+str(filename)+str(i)+'.jpg',frame)
		i += 1
	
	cap.release()
	cv2.destroyAllWindows()
	print("finished converting video to frames...")
	
	
	##############
	# convert to painting
	print("continuing to convert the frames to painting...")
	painting_array = []
	size = ()
	printProgressBar(0, i, prefix = 'Progress:', suffix = 'Complete', length = 50)
	for j in range(0, i):
		## actual conversion
		img = cv2.imread('temp'+str(filename)+'/temp'+str(filename)+str(j)+'.jpg')
		
		if args.mode == "oil":
			res = toOilPainting(img)
		elif args.mode == "point":
			res = toPointillismPainting(img)
		# elif args.mode == "ascii":
			# ## special case cuz the input is filename, so whatever
			# res = toASCIIArt('temp'+str(filename)+'/temp'+str(filename)+str(j)+'.jpg')
		##
		height, width, layers = img.shape
		size = (width,height)
		painting_array.append(res)
		#cv2.imwrite('temp'+str(i)+'conv.jpg', img)
		printProgressBar(j+1, i, prefix = 'Progress:', suffix = 'Complete', length = 50)
	print("finished converting the frames to painting")
	
	print("continuing to convert the painting frames back to video...")
	video_output = cv2.VideoWriter('painted_'+str(filename)+"_"+str(args.mode)+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), target_fps, size)
	printProgressBar(0, len(painting_array), prefix = 'Progress:', suffix = 'Complete', length = 50)
	for j in range(len(painting_array)):
		video_output.write(painting_array[j])
		printProgressBar(j+1, len(painting_array), prefix = 'Progress:', suffix = 'Complete', length = 50)
	video_output.release()
	print("all finished!")

except Exception as e:
	print("\nEncountered error!")
	print(e)
	traceback.print_exc()

##############
# cleaning up
finally:
	filelist = glob.glob(os.path.join(os.getcwd(), 'temp'+str(filename)+'/*.jpg'))
	for f in filelist:
		os.remove(f)
	os.rmdir("temp"+str(filename))
	print("removed temporary images")