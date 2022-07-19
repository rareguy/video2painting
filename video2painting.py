def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import cv2
import glob
import os
import argparse
import math
import sys
import triangler

from tqdm import tqdm
from pointillism import *
import linedraw
import traceback
from PIL import Image, ImageColor
import moviepy.video.io.ImageSequenceClip
import natsort
import concurrent.futures

###
# cleaning up warnings that littered the output
import warnings
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#######################
### UTILITY FUNCTIONS
#######################

def convert2png(svg_file, png_file, background, resolution = 300):
    from wand.api import library
    import wand.color
    import wand.image

    with open(svg_file, "r") as svg_file:
        with wand.image.Image() as image:
            with wand.color.Color(background) as background_color:
                library.MagickSetBackgroundColor(image.wand, 
                                                 background_color.resource) 
            svg_blob = svg_file.read().encode('utf-8')
            image.read(blob=svg_blob, resolution = resolution, background=background)
            png_image = image.make_blob("png32")

    with open(png_file, "wb") as out:
        out.write(png_image)

def convert2video(src_folder, target, target_fps):
	image_folder = src_folder
	image_files = natsort.natsorted([os.path.join(image_folder,img)
               for img in os.listdir(image_folder)
               if img.endswith(".png")])
	clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=target_fps)
	clip.write_videofile(target)

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

#####################
### MODES FUNCTIONS
#####################

list_of_mode = ['oil', 'point', 'linedraw', 'triangler']

def toOilPainting(img):
	return cv2.xphoto.oilPainting(img, 8, 1)

def toOilPaintingConcurrent(filename):
	pngname = filename[:-4]+"_pngout.png"
	img = cv2.imread(filename)
	res = cv2.xphoto.oilPainting(img, 8, 1)
	cv2.imwrite(pngname, res)

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

def toPointillismPaintingConcurrent(filename):
	img = cv2.imread(filename)
	pngname = filename[:-4]+"_pngout.png"
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

	cv2.imwrite(pngname, res)

def toLinedraw(filedir, filename):
	lines = linedraw.sketch(os.path.join(filedir, filename), filename)
	pngname = filename[:-4]+"_pngout.png"
	pngname_dir = os.path.join("output", pngname)
	png_white_bg_name = filename[:-4]+"_pngwout.png"
	png_white_bg_name_dir = os.path.join("output", png_white_bg_name)
	convert2png("output/"+filename+".svg", pngname_dir)
	
	image=Image.open(pngname_dir)
	non_transparent=Image.new('RGB',(3840,2160),(0,0,0))
	non_transparent.paste(image,(0,0),image)
	non_transparent.save(png_white_bg_name_dir)
	os.remove(pngname_dir)
	res = cv2.imread(png_white_bg_name_dir)
	return res

def toLinedrawConcurrent(filename, color, background):
	pngname = filename[:-4]+"_pngout.png"
	png_white_bg_name = filename[:-4]+"_pngwout.png"
	svgout = filename[:-4]+".svg"
	lines = linedraw.sketch(filename, color, svgout)
	
	convert2png(svgout, pngname, background)
	backgroundtuple = ImageColor.getcolor(background, "RGB")

	image=Image.open(pngname)
	non_transparent=Image.new('RGB',(3840,2160),backgroundtuple)
	non_transparent.paste(image,(0,0),image)
	non_transparent.save(png_white_bg_name)
	os.remove(pngname)

def toTrianglerConcurrent(filename):
	pngname = filename[:-4]+"_pngout.png"
	triangler_instance = triangler.Triangler()
	triangler_instance.convert_and_save(filename, pngname)

#########
### MAIN
#########
def main():
	parser = argparse.ArgumentParser(description='this script will convert your normal video to painting-style video. example usage: video2painting.py oil video.mp4')
	parser.add_argument('mode', metavar='mode', type=str, help='the video conversion mode ('+ ", ".join(list_of_mode)+ ")")
	parser.add_argument('source_video', metavar='source', type=str, help='the video filename (currently only mp4) put the source ONLY IN CURRENT DIRECTORY, if you use exact path it will bug out')
	parser.add_argument('--fps', metavar='fps', type=int, nargs="?", help='custom fps to make the visual looks cooler', default=0)
	parser.add_argument('--color', metavar='color', type=str, nargs="?", help='(linedraw only) (RGB ONLY) use certain line color, FORMAT: #XXXXXX', default="#000000")
	parser.add_argument('--background', metavar='background', type=str, nargs="?", help='(linedraw only) (RGB ONLY) use certain background color, FORMAT: #XXXXXX', default="#FFFFFF")
	args = parser.parse_args()

	if not (args.mode in list_of_mode):
		print(args.mode, "is not available")
		assert False


	inputfile = args.source_video
	print("processing", inputfile, "...")	
	try:
		target_fps = 0
		# Opens the Video file
		cap= cv2.VideoCapture(inputfile)
		filename = inputfile[:-4]
		fps = cap.get(cv2.CAP_PROP_FPS)
		if args.fps > 0 and args.fps <= fps:
			target_fps = args.fps
		else:
			target_fps = fps
		color = args.color
		background = args.background
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
		if args.mode == "linedraw":
			print("fair warning, this will use your entire CPU pool to make the conversion feasibly fast.\npress ctrl+c if you're not ready to use this yet.")
			all_raw_image = glob.glob('temp'+str(filename)+'/temp'+str(filename)+'*.jpg')
			len_all_raw_image = len(all_raw_image)
			with tqdm(total=len_all_raw_image) as pbar:
				with concurrent.futures.ProcessPoolExecutor() as executor:
					# special case cuz the input is filename, so whatever
					futures = {executor.submit(toLinedrawConcurrent, arg, color, background): arg for arg in all_raw_image}
					for future in concurrent.futures.as_completed(futures):
						pbar.update(1)
		elif args.mode == "oil":
			print("fair warning, this will use your entire CPU pool to make the conversion feasibly fast.\npress ctrl+c if you're not ready to use this yet.")
			all_raw_image = glob.glob('temp'+str(filename)+'/temp'+str(filename)+'*.jpg')
			len_all_raw_image = len(all_raw_image)
			with tqdm(total=len_all_raw_image) as pbar:
				with concurrent.futures.ProcessPoolExecutor() as executor:
					# special case cuz the input is filename, so whatever
					futures = {executor.submit(toOilPaintingConcurrent, arg): arg for arg in all_raw_image}
					for future in concurrent.futures.as_completed(futures):
						pbar.update(1)
		elif args.mode == "point":
			print("fair warning, this will use your entire CPU pool to make the conversion feasibly fast.\npress ctrl+c if you're not ready to use this yet.")
			all_raw_image = glob.glob('temp'+str(filename)+'/temp'+str(filename)+'*.jpg')
			len_all_raw_image = len(all_raw_image)
			with tqdm(total=len_all_raw_image) as pbar:
				with concurrent.futures.ProcessPoolExecutor() as executor:
					# special case cuz the input is filename, so whatever
					futures = {executor.submit(toPointillismPaintingConcurrent, arg): arg for arg in all_raw_image}
					for future in concurrent.futures.as_completed(futures):
						pbar.update(1)
		elif args.mode == "triangler":
			print("fair warning, this will take long time because the algorithm is RAM heavy which is not ideal for CPU multiprocessing (which is default for other mode)")
			all_raw_image = glob.glob('temp'+str(filename)+'/temp'+str(filename)+'*.jpg')
			len_all_raw_image = len(all_raw_image)
			with tqdm(total=len_all_raw_image) as pbar:
				for image in all_raw_image:
					toTrianglerConcurrent(image)
					pbar.update(1)
		pbar.close()
		print("finished converting the frames to painting")
		print("continuing to convert the painting frames back to video...")
		convert2video("temp"+str(filename), 'painted_'+str(filename)+"_"+str(args.mode)+'.mp4', target_fps)
		print("all finished!")

	except Exception as e:
		print("\nEncountered error!")
		print(e)
		traceback.print_exc()

	##############
	# cleaning up
	finally:
		filelist = glob.glob(os.path.join(os.getcwd(), 'temp'+str(filename)+'/*.jpg'))
		filelist2 = glob.glob(os.path.join(os.getcwd(),'temp'+str(filename)+'/*.svg',))
		filelist3 = glob.glob(os.path.join(os.getcwd(), 'temp'+str(filename)+'/*.png'))
		for f in filelist:
			os.remove(f)
		for f in filelist2:
			os.remove(f)
		for f in filelist3:
			os.remove(f)
		os.rmdir("temp"+str(filename))
		print("removed temporary images")

if __name__ == "__main__":
	main()