from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

cam = cv2.VideoCapture(0)

layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("Loading EAST text detector...")
net = cv2.dnn.readNet("frozen_east_text_detection.pb")


def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

def recognize(im):
    orig = im.copy()
    (origH, origW) = im.shape[:2]
    (newW, newH) = (320, 320)
    rW = origW / float(newW)
    rH = origH / float(newH)
    im = cv2.resize(im, (newW, newH))
    (H, W) = im.shape[:2]
    
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(im, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    results = list()
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        im = orig[startY:endY, startX:endX]
             
        
        #recognizer starts here
        dX = int((endX - startX) * 0.03)
        dY = int((endY - startY) * 0.03)
        
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        im2 = orig[startY:endY, startX:endX]
        
        text = pytesseract.image_to_string(im2,config='--psm 10 --oem 3')
        
        cv2.putText(orig,text,(startX, startY-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0, 255, 0),1)
        
        
    return orig
            

def main():
    while True:
        ret,frame = cam.read()
        frame = recognize(frame)
        cv2.imshow("detection !",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
main()