# video2painting

A short script to convert any video supported by OpenCV Python (without audio) to artsy styled video like stop-motion painting etc. Currently uses three main mode: `oil`, `point`, and `linedraw`.

Written in Python3.
Tested in Python3.7 in Windows

How to use:
```
python video2painting.py -h
usage: video2painting.py [-h] [--fps [fps]] mode source

this script will convert your normal video to painting-style video. example usage: video2painting.py oil video.mp4

positional arguments:
  mode         the video conversion mode (oil, point, ascii)
  source       the video filename (currently only mp4)

optional arguments:
  -h, --help   show this help message and exit
  --fps [fps]  custom fps to make the visual looks cooler
```

I noticed that with higher FPS, the video doesn't look any better than downgraded video quality on youtube, so I made an option to have custom fps that should be *lower* than the original. More or zero will be counted as default, the same as original video. I recommend to have around 12 FPS so it will feel like stop-motion rather than low resolution video.

Requirements:
- opencv2-python
- opencv2-contrib-python
- progressbar2
- makeasciiart
- also read requirements.txt

Original author of each mode:
`oil` --- opencv2-contrib

`point` --- [Pointillism](https://github.com/matteo-ronchetti/Pointillism) by Matteo Ronchetti (MIT License)

`linedraw` --- [linedraw](https://github.com/LingDong-/linedraw) by Lingdong Huang (MIT License)

## Modes and its weird behaviour
### Oil
Working as intended and CPU intensive.
### Point
Working as intended. Took really long to convert.
### Linedraw
It works but will always output in 4K due to the library is outputting SVG rather than image of `linedraw`. Setting the "frame" without the resolution at start will make the output image size vary slightly (usually the width), and to compile frames to video requires the images to be in the exact same size. So to use this I suggest to use a 16:9 aspect ratio to make sure it doesn't crop or looks weird. I haven't thought of any solution other than doing some algorithm to find the max height or width for the collection of frames which require me some time.

### Unusual stuffs
- I made this in python 3.7 with conda and for some reason using base version it's not working at all. (Some Windows library issue)
- `makeascii` option is still buggy because weird behaviour of the library
- Any new video option ideas are welcome