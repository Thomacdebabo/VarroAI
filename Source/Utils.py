import cv2
import numpy as np
import datetime
import os

DEBUG=0

from tensorflow.keras.models import Model, load_model
def getPicfromKeypoint(kp, img, size=(50,50)):
    #Takes a keypoint and an image and returns a cutout of the image around the keypoint with a given size
    size = np.array(size)
    p = np.array(kp.pt)
    p1 = (p - size).astype("int")
    p2 = (p + size).astype("int")
    return (img[p1[1]:p2[1],p1[0]:p2[0],:])

def KeypointsToPictures(kps, img):
    pics = []
    for kp in kps:
        pic = getPicfromKeypoint(kp, img)
        if(pic.shape == (100,100,3)):
            pics.append(pic)
        else: 
            if(DEBUG): print("skipped pic because of wrong shape")    
    return pics

def evaluateVarroaHits(pics, dir):
    # function to evaluate all varroa samples manually
    #it will label the pictures and store them in the given directory (dir) and return the amount of varroa, false positives and skipped pictures
    
    #to use the tool press: 
    #"a" to label a sample as varroa, 
    #"d" to tag it as a falsep-ositive,
    #any other key to skip the sample
    varroa = 0
    false_pos = 0
    skipped = 0
    
    for pic in pics:
        try:
            cv2.imshow("image", cv2.resize(pic, (800,800)))
            keypressed =cv2.waitKey(0)
            if keypressed == ord('a'):
                print("varroa")
                varroa+=1
                cv2.imwrite(dir+r"\V_" + datetime.datetime.now().strftime("%H_%M_%S")+"_"+str(varroa)+".jpg", pic)
            elif  keypressed== ord('d'):
                print("falsepositive")
                false_pos+=1
                cv2.imwrite(dir+r"\F_"+ datetime.datetime.now().strftime("%H_%M_%S")+"_"+str(false_pos)+".jpg", pic)
            else:
                print("skipped")
                skipped+=1
        except:
            if(DEBUG): print("Image could not load")
    print("Varroa: "+str(varroa)+" False positives: "+ str(false_pos) + " Skipped: "+  str(skipped))
    return varroa, false_pos, skipped

def threshholding(img):#takes image and returns thresholded a thesholded image with only the important sections left
                        #It works by using 2 stages of thresholding, one adaptive and one static
    BGmask = getBackgroundMask(img)
    kernel = np.ones((3,3))
    
    gray = img[:,:,2]
    gray = cv2.medianBlur(gray,5)
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    thr = np.maximum(cv2.erode(thr, kernel),BGmask)
    
    mask = np.stack((thr,thr,thr), axis=-1)
    ada = np.maximum(mask, img)
    first = np.maximum(thr, gray)
    
    ret, second = cv2.threshold(first, 130, 255, cv2.THRESH_BINARY )
    mask_2 = np.stack((second,second,second), axis=-1)
    
    return np.maximum(mask_2, img)

def findblobs(img): #finds all blobs in a given Image
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
     
    # Change thresholds
    params.minThreshold = 120;
    params.maxThreshold = 200;
     
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 20
     
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
     
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.8
     
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
     
    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)
    image = threshholding(img)
    # Detect blobs.
    keypoints = detector.detect(image)
     
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return im_with_keypoints, keypoints, keypoints.__len__()


def getBackgroundMask(img): #Returns a Mask 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5),  cv2.BORDER_DEFAULT)
    ret, img = cv2.threshold(img,150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((10,10))
    img = cv2.erode(img, kernel)
    img = cv2.dilate(img, kernel, iterations=4)
    img = cv2.erode(img, kernel, iterations=4)
    return 255-img


#Main Code


def useVarroAI(pic, model_path =r"Models/VarroaI_2.hdf5" ): #Uses an AI approach to evaluate the varroa Samples.
    #It takes all samples in one go and returns the predictions for each and everyone of the samples
    pic = np.array(pic).astype('float32')/255.
    model = load_model(model_path)
    if(pic.shape[-3:]==(100,100,3)):
        return model.predict(pic)
    else:
        print("wrong shape of picture")

def evalVarroaAI(pred, split_up=0.9, split_down=0.3): #splits the predictions into varroa / falsepositives /unsure  
    varroa = pred[pred >split_up].size
    unsure = pred[pred <= split_up]
    unsure = unsure[unsure >= split_down].size
    falsepositives = pred[ pred < split_down].size

    return {"Varroa": varroa, "False Positives": falsepositives,"Unsure": unsure} #return is a dictionary to avoid mix-ups
def detect_Varroa(img_path):
    img = cv2.imread(img_path, 1)
    dst, kps, count = findblobs(img)
    VarroaImgs = KeypointsToPictures(kps, img)
    AI_Pred = evalVarroaAI(useVarroAI(VarroaImgs))
    return AI_Pred
def batchProcessing(dir): #iterates over all pictures in a directory (including subdirectories) and detects the varroa using preprocessing and AI
    AI_predictions = []
    for path, subdirs, files in os.walk(dir):
        for name in files:
            img_path = path+"\\"+name
            AI_Pred = detect_Varroa(img_path)
            
            AI_predictions.append((name,AI_Pred))
            
            print(name + ": " + str(AI_Pred))

    return AI_predictions
def Img_to_Dataset(img_dir, dataset_dir): #evaluate hits by hand and add them to the dataset
    im_orig = cv2.imread(img_dir)
    
    #mask = np.stack((image,image,image), axis=-1)
    dst, kps, count = findblobs(im_orig)
    VarroaImgs = KeypointsToPictures(kps, im_orig)
    
    Varroa, False_Positives, Skipped = evaluateVarroaHits(VarroaImgs, dataset_dir)

