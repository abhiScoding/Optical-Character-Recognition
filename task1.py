# OpenCV 3.4.5
"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
from re import template
from unittest import result
import cv2
import numpy as np
import collections 

def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=5000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    charDict = enrollment(characters)

    lableDict, bBox = detection(test_img)
    
    recognitionDict = recognition(charDict,lableDict)

    recognitionList = []
    for lable, chr in recognitionDict.items():
        resultDict = {}
        x = bBox[lable][0]
        y = bBox[lable][1]
        w = bBox[lable][2] - bBox[lable][0] + 1
        h = bBox[lable][3] - bBox[lable][1] + 1
        resultDict["bbox"] = [x,y,w,h]
        resultDict["name"] = chr
        recognitionList.append(resultDict)

    # print('printing recognition list!')
    # print(recognitionList)

    return recognitionList
    # returns [{'bbox':[x,y,w,h],'name':'a'},...]
    # raise NotImplementedError

# binarizing img: background pixcel=0, character pixel=1
def binarize(img):
    rws, cls = len(img), len(img[0])
    img = np.ravel(img)
    for i in range(len(img)):
        if img[i]>=100:
            img[i]= 0
        else:
            img[i]= 1
    bin_img = img.reshape(rws,cls)
    return bin_img

# binarize lables in lableDict: back=0, chr=255
def binLables(img):
    rws, cls = len(img), len(img[0])
    img = np.ravel(img)
    for i in range(len(img)):
        if img[i] == 0:
            img[i]= 0
        else:
            img[i]= 255
    bin_img = img.reshape(rws,cls)
    return bin_img

# create bounding box
def createBB(labledImg): 
    bbox = {} 
    
    rows, cols = len(labledImg), len(labledImg[0])

    maxLableVal = np.amax(labledImg)

    for lable in range(1,maxLableVal+1):
        xmin,ymin,xmax,ymax = np.inf,np.inf,0,0
        for x in range(cols):
            for y in range(rows):
                if labledImg[y][x] == lable:
                    if x<=xmin:
                        xmin = x
                    if y<=ymin:
                        ymin = y
                    if x>=xmax:
                        xmax = x
                    if y>=ymax:
                        ymax = y
        bbox[lable] = [xmin,ymin,xmax,ymax]
    return bbox

# display bounding box
def displyBbox(labledImage, bBox):

    for lable in bBox:
        xmn = bBox[lable][0]
        ymn = bBox[lable][1]
        xmx = bBox[lable][2]
        ymx = bBox[lable][3]

        for x in range(xmn,xmx+1):
            labledImage[ymn][x] = 200
            labledImage[ymx][x] = 200
        for y in range(ymn, ymx):
            labledImage[y][xmn] = 200
            labledImage[y][xmx] = 200

    return labledImage

# extract characters from labled img and stores into dictionary
def extChar(labledImg,bBox):

    # bBox: {'1':[xmin,ymin,xmax,ymax],...}

    y_min = []

    # print(bBox)
    for lable, coor in bBox.items():
        y_low = coor[3]

        if y_low not in y_min:
            y_min.append(y_low)

    oymin = sorted(y_min)

    # print('printing oymin')
    # print(oymin)

    oLables = []
    cols = len(labledImg[0])

    for y in oymin:
        for x in range(cols):
            lableVal = labledImg[y][x]
            if lableVal !=0:
                if lableVal not in oLables:
                    oLables.append(lableVal)

    # print('printing order lables',len(oLables))
    # print(oLables)

    obBox = {}
    for o_lable in oLables:
        obBox[o_lable] = bBox[o_lable]

    # print('printing order bBox')
    # print(obBox)

    olableDict = {}

    for labl, coord in obBox.items():
        x1, y1, x2, y2 = coord[0], coord[1], coord[2], coord[3]
        olableDict[labl] = labledImg[y1-1:y2+2,x1-1:x2+2]

    # print(olableDict[2])
    
    return olableDict
    # returns lable dict in english reading order: {lable:img,...}

# connected componant labling to lable each characters
def CCL(img):
    width, hight = len(img), len(img[0])
    seen = set()
    lable = 0

    def brFirstSearch(w,h):
        dq = collections.deque()

        seen.add((w,h))
        dq.append((w,h))
        while dq:
            wth, hgt = dq.popleft()

            neighbors = [[1,0],[-1,0],[0,1],[0,-1]]
            for inc_w,inc_h in neighbors:
                w,h = wth + inc_w, hgt+inc_h
                if (w in range(width) and h in range(hight) and img[w][h] == 1 and (w,h) not in seen):
                    dq.append((w,h))
                    seen.add((w,h))
                    img[w][h] = lable

    for w in range(width):
        for h in range(hight):
            if img[w][h] == 1 and (w,h) not in seen:
                lable += 1
                brFirstSearch(w,h)
                img[w][h] = lable
    return img 

def enrollment(characters):
    
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    print("Enrollment starts!")

    #binarizing characters backg=0, char=1
    for character in range(len(characters)):
        binarize(characters[character][1])
    
    charDict ={}
    # extrecting characters form images and storing in dict
    for idx in range(len(characters)):
        charDict[characters[idx][0]] = extChar(characters[idx][1],createBB(characters[idx][1]))[1]

    # multiplying each image in charDict by 255
    for cr, imge in charDict.items():
        charDict[cr] = imge*255
  
    # extracting features using canny edge detector
    for key, val in charDict.items():
        charDict[key] = cv2.Canny(charDict[key],100,200)

    # print('showing charDict')
    # print(charDict)
    # show_image(charDict['e'],0)
    print('Enrollment done!')
    return charDict
    # returns : {'2':cannyArry,....} backg = 0 , edge = 255
    # raise NotImplementedError

def detection(testImg):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    print('detection starts!')

    # binarize testImg
    bin_img = binarize(testImg)

    # lableing binarized img
    labeledImg = CCL(bin_img)
    # print(labeledImg)

    # creating bbox on labled img
    bBox = createBB(labeledImg)
    # print(bBox)
    # returns {1: [92, 4, 105, 38],.....}

    # display labledImg with bbox
    # boxdImg = displyBbox(labeledImg,bBox)
    # show_image(boxdImg,4000)
        
    # extrecting characters from labled image and stores into dictionary
    lableDict = extChar(labeledImg,bBox)
  
    
    # binarize labled images and multiplyig with 255
    for cr, ig in lableDict.items():
        lableDict[cr]= binLables(ig)
    

    # show_image(lableDict[21],0)

    print('Number of characters detected: ',len(lableDict))
    # print(lableDict)
    print('detection ends!')
    return lableDict,bBox
    # returns: lableDict = {lable:imgArray,....} backg=0, chr=255
              # bbox = {1: [92, 4, 105, 38],.....}

    # raise NotImplementedError

def recognition(charDict,lableDict):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    print('recognition starts!')

    recognitionDict = {}
    totalIdentified = 0

    errorMinList = []
    for lable, lablEimage in lableDict.items():
        lableEdges = cv2.Canny(lablEimage,100,200)
        errorMin = np.inf

        # implementing SSD for feature matching
        for chr, chrEdges in charDict.items():
            lableWidth = len(lablEimage[0])
            lableHight = len(lablEimage)
            templateEdges = binLables(cv2.resize(chrEdges,(lableWidth,lableHight)))
            # print(templateEdges)
            diff = lableEdges - templateEdges
            sqr = np.multiply(diff,diff)
            error = np.sum(sqr)/(lableWidth*lableHight)

            if error<errorMin:
                errorMin = error
                chract = chr

        errorMinList.append(errorMin)
        recognitionDict[lable] = "UNKNOWN"
        if errorMin< 0.27:
            recognitionDict[lable] = chract
            totalIdentified += 1
    
    # print(len(errorMinList))
    # print('')
    # print(errorMinList)
    # print('')
    print('')
    print('total characters in test image: ', len(recognitionDict))
    print('')
    print(recognitionDict)
    print('')
    print('num of characters recognised: ',totalIdentified)
    print('recognition ends!')

    return recognitionDict
    #returns: {...., 26:'e',.......}
    # raise NotImplementedError




def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])
    
    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
