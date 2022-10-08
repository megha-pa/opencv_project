from distutils.command.config import config
import cv2
import numpy as np
import time
np.random.seed(20)
class Detector:
    def __init__(self,videoPath,configPath,modelPath,classPath):
        self.videoPath=videoPath
        self.configPath=configPath
        self.modelPath=modelPath
        self.classesPath=classPath


        self.net=cv2.dnn_DetectionModel(self.modelPath,self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)
        self.readClasses()


    def readClasses(self):
        with open(self.classesPath,'r')as f:
            self.classesList=f.read().splitlines()
        self.classesList.insert(0,'__Background__')
        self.colorList=np.random.uniform(low=0,high=255,size=(len(self.classesList),3))
        print(self.classesList)

    def onVideo(self):
        cap=cv2.VideoCapture(self.videoPath)

        if(cap.isOpened()==False):
            print("Error loading File")
            return 

        (succes,image)=cap.read()

        while succes:
            classLabelIDs,confidencde,bboxs=self.net.detect(image,confThreshold=0.5)
            bboxs=list(bboxs)
            confidencdes=list(np.array(confidencde).reshape(1,-1)[0])
            confidencdes=list(map(float,confidencdes))
            bboxsIdx=cv2.dnn.NMSBoxes(bboxs,confidencdes,score_threshold=0.5,nms_threshold=0.3)
            
            if len(bboxsIdx)!=0:

                for i in range(0,len(bboxsIdx)):
                    bbox=bboxs[np.squeeze(bboxsIdx[i])]
                    classConfidence=confidencdes[np.squeeze(bboxsIdx[i])]
                    classLabelID=np.squeeze(classLabelIDs[np.squeeze(bboxsIdx[i])])
                    classLabel=self.classesList[classLabelID]
                    classColors=[int(c) for c in self.colorList[classLabelID]]

                    displayText="{}:{:.2f}".format(classLabel,classConfidence)
                    x,y,w,h=bbox

                    cv2.rectangle(image,(x,y),(x+w,y+h),color=classColors,thickness=1)
                    cv2.putText(image,displayText,(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,classColors,2)
                    

                    lineWidth=min(int(w*0.3),int(h*0.3))
                    cv2.line(image,(x,y),(x+lineWidth,y),classColors,thickness=5)
                    cv2.line(image,(x,y),(x,y+lineWidth),classColors,thickness=5)
                    cv2.line(image,(x+w,y),(x+w-lineWidth,y),classColors,thickness=5)
                    cv2.line(image,(x+w,y),(x+w,y+lineWidth),classColors,thickness=5)

                    cv2.line(image,(x,y+h),(x+lineWidth,y+h),classColors,thickness=5)
                    cv2.line(image,(x,y+h),(x,y+h-lineWidth),classColors,thickness=5)
                    cv2.line(image,(x+w,y+h),(x+w-lineWidth,y+h),classColors,thickness=5)
                    cv2.line(image,(x+w,y+h),(x+w,y+h-lineWidth),classColors,thickness=5)


            cv2.imshow("result",image)

            key=cv2.waitKey(1) & 0xFF
            if key==ord("q"):
                break
            (succes,image)=cap.read()
        cv2.destroyAllWindows()