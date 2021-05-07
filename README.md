# video2painting

A short script to convert any video supported by OpenCV Python (without audio) to artsy styled video like stop-motion painting etc. Currently uses three main mode: `oil`, `point`, and `ascii`.

Written in Python3.

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

The one bug I also noticed is by reducing the FPS, the video result will be longer in proportion of FPS reduction. Haven't thought about the solution, but I'll get into it in the future.

Requirements:
- opencv2-python
- opencv2-contrib-python
- progressbar2
- makeasciiart

Original author of each mode:
`oil` --- opencv2-contrib
`point` --- [Pointillism](https://github.com/matteo-ronchetti/Pointillism) by Matteo Ronchetti (MIT License)
`ascii` --- [makeasciiart](https://pypi.org/project/makeasciiart/) by SSS (MIT License)
