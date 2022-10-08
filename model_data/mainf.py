from detector import *
import os

def main():
    videoPath='cars.mp4'
    configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    modelPath='frozen_inference_graph.pb'
    classPath='coco.names'

    detector=Detector(videoPath,configPath,modelPath,classPath)
    detector.onVideo()

if __name__ == '__main__':
    main()