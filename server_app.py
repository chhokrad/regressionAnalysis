'''
Created on Nov 19, 2016

@author: ajaychhokra
'''
import cv2

class VideoObjectError(Exception): pass

class FrameGrabber(object):
    '''
    classdocs
    '''


    def __init__(self, videoFile = None, fromCamera = False):
        '''
        Constructor
        '''
        
        if videoFile is None and not fromCamera :
            self.myVideObject = cv2.VideoCapture()
        elif videoFile is None and fromCamera:
            self.myVideObject = cv2.VideoCapture(0)
            self.myVideObject.open(0)
        else:
            self.myVideObject = cv2.VideoCapture(videoFile)
            self.myVideObject.open(videoFile)
        
        #self.myVideObject.set('CV_CAP_PROP_FOURCC',cv2.CV_FOURCC('M','P','G','4'))
        if not self.myVideObject.isOpened():
            raise VideoObjectError('Initialization Failed')
        else:
            self.fps = self.myVideObject.get(cv2.cv.CV_CAP_PROP_FPS)
            self.startFrame = True
            self.startsending()
        
    def startsending(self):
        while self.startFrame :
            self.startFrame, frame = self.myVideObject.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(int(1000/self.fps))
        self.myVideObject.release()
