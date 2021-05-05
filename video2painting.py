import cv2
import glob
import os
import time
import argparse
# Print iterations progress
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

parser = argparse.ArgumentParser(description='this script will convert your normal video to painting-style video')
parser.add_argument('source_video', metavar='source', type=str, nargs='+', help='the video filename (currently only mp4)')
args = parser.parse_args()

inputfile = args.source_video[0]

try:
	# img = cv2.imread('img.jpg')
	# res = cv2.xphoto.oilPainting(img, 7, 1)

	# Opens the Video file
	cap= cv2.VideoCapture(inputfile)
	filename = inputfile[:-4]
	fps = cap.get(cv2.CAP_PROP_FPS)
	print("FPS:", fps)
	i = 0
	print("converting video to frames...")
	try:
		os.mkdir("temp")
	except:
		print("i think there is already a folder named temp, pls delete it")
		assert False
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == False:
			break
		cv2.imwrite('temp/temp'+str(filename)+str(i)+'.jpg',frame)
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
		img = cv2.imread('temp/temp'+str(filename)+str(j)+'.jpg')
		res = cv2.xphoto.oilPainting(img, 8, 1)
		height, width, layers = img.shape
		size = (width,height)
		painting_array.append(res)
		#cv2.imwrite('temp'+str(i)+'conv.jpg', img)
		printProgressBar(j+1, i, prefix = 'Progress:', suffix = 'Complete', length = 50)
	print("finished converting the frames to painting")
	
	print("continuing to convert the painting frames back to video...")
	video_output = cv2.VideoWriter('painted_'+str(filename)+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
	printProgressBar(0, len(painting_array), prefix = 'Progress:', suffix = 'Complete', length = 50)
	for j in range(len(painting_array)):
		video_output.write(painting_array[j])
		printProgressBar(j+1, len(painting_array), prefix = 'Progress:', suffix = 'Complete', length = 50)
	video_output.release()
	print("all finished!")

except Exception as e:
	print("\nencountered error!")
	print(e)

##############
# cleaning up
finally:
	filelist = glob.glob(os.path.join(os.getcwd(), "temp/*.jpg"))
	for f in filelist:
		os.remove(f)
	os.rmdir("temp")
	print("removed temporary images")
