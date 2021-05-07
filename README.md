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

Original author of each mode:
`oil` --- opencv2-contrib
`point` --- [Pointillism](https://github.com/matteo-ronchetti/Pointillism) by Matteo Ronchetti (MIT License)
`ascii` --- [Rune](https://www.learnpythonwithrune.org/ascii-art-of-live-webcam-stream-with-opencv/)
