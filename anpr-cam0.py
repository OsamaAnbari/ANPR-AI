# ---------------------- Import ----------------------

import os
import sys
from PIL import Image, ImageTk
from IPython.display import display
import cv2
import numpy as np
import pathlib
import csv
import datetime
import matplotlib.pyplot as plt
import tkinter as tk

import tensorflow as tf
from tensorflow import keras

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import tkinter
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

# ---------------------- Packages ----------------------
#
# opencv-python == 4.7.0
# numpy == 1.24.2
# tensorflow == 2.10.1
#
# ---------------------- Setup Object Detection ----------------------
#
# - Download protobuf and add <bin> to environments
# - Clone Tensorflow Garden
# - From within TensorFlow/models/research/ (protoc object_detection/protos/*.proto --python_out=.)
# - From within TensorFlow/models/research/ (cp object_detection/packages/tf2/setup.py .) (python -m pip install .)




# ---------------------- Detecting ----------------------------------------------------------------------

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
    return output_dict

# ---------------------- Detecting Real Time ----------------------------------------------------------------------

def show_inference(model, frame):
    #take the frame from webcam feed and convert that to array
    image_np = np.array(frame)
    # Actual detection.
    
    result_dict = run_inference_for_single_image(model, image_np)
    try:
        boxing(image_np,result_dict)
    except:
        pass
    
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        result_dict['detection_boxes'],
        result_dict['detection_classes'],
        result_dict['detection_scores'],
        category_index,
        instance_masks=result_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=2)
    
    return(image_np)

def real_time():
    global video_capture
    global camera_NO
    video_capture = cv2.VideoCapture(camera_NO)
    while True:
        ret,frame = video_capture.read()
        Imagenp=show_inference(model_SSD, frame)

        img_update = ImageTk.PhotoImage(Image.fromarray(Imagenp))
        paneeli_image.configure(image=img_update)
        paneeli_image.image=img_update
        paneeli_image.update()

        if not ret:
            print("failed to grab frame")
            break

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")

            video_capture.release()
            cv2.destroyAllWindows()
            break
        #cv2.imshow('Automatic Number Plate Recognition', cv2.resize(Imagenp, (800,600)))
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    #video_capture.release()
    
# ---------------------- Detecting From Image ----------------------------------------------------------------------

def model_show(model, image_path):
    image = np.array(Image.open(image_path))
    
    result_dict = run_inference_for_single_image(model,image)
    boxing(image,result_dict)
    
    vis_util.visualize_boxes_and_labels_on_image_array(
      image,
      result_dict['detection_boxes'],
      result_dict['detection_classes'],
      result_dict['detection_scores'],
      category_index,
      instance_masks=result_dict.get('modified_detection_masks', None),
      use_normalized_coordinates=True,
      line_thickness=2)
    
    display(Image.fromarray(image).show())

# ---------------------- OCR ----------------------------------------------------------------------

def find_contours(dimensions, img) :
    
    ii = img.copy()

    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
        
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            #plt.imshow(ii, cmap='gray')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
    
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

# Find characters in the resulting images
def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/1,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

def fix_dimension(img): 
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = img
    return new_img

def OCR_CNN(char, model):
    
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c
        
    output = []
    for i,ch in enumerate(char):
        img_ = cv2.resize(ch, (28,28))
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3)
        y_ = model.predict(img, verbose = 0)[0];
        z=0
        for r in range(35):
            if (y_[r]==1.0):
                z=r
        character = dic[z]
        output.append(character)
        
    plate_number = ''.join(output)
    
    return plate_number

# ---------------------- Boxing and do_OCR ----------------------------------------------------------------------

# -------------
i = 1
OCR = 0
OCR_Past = 0
# -------------

def boxing(image,output_dict):
    scores = list(filter(lambda x: x> detection_threshold, output_dict['detection_scores']))
    boxes = output_dict['detection_boxes'][:len(scores)]
    classes = output_dict['detection_classes'][:len(scores)]
    width = image.shape[1]
    height = image.shape[0]
    region = 0;
    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        region = Image.fromarray(region)
    
    do_OCR(region)

def do_OCR(region):
    if (region):
        global OCR
        global OCR_Past
        global category_index
        img = np.array(region)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        char = segment_characters(img)
        
        if (len(char) >= 3):
            OCR_Past_Past = OCR_Past
            OCR_Past = OCR
            OCR = OCR_CNN(char, model_CNN)
            status = ' '#check_status(OCR)
            try:
                category_index = {1: {'id': 1, 'name': OCR+" "+status }}
            except:
                category_index = {1: {'id': 1, 'name': OCR}}
            if True : #((OCR == OCR_Past) and (OCR != OCR_Past_Past)):
                display(region)
                save_res(img, OCR)
                print(OCR)
        else:
            category_index = {1: {'id': 1, 'name': ' '}}

# ---------------------- Save and Check Database ----------------------------------------------------------------------

def save_res(img, O):
    global i
    img_name = Detected_Plates_Path + '/' + f'{i}' + '-' + O + '.jpg' 
    cv2.imwrite(img_name, img)
    date = datetime.datetime.now()
    i = i + 1
    
    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([date, O])

# ---------------------- Configuration ----------------------------------------------------------------------

SSD_Model_Path = "D:/Projects/Machine Learning/anpr-rasp/data/my_model_SSD/saved_model" #"data/my_model_SSD/saved_model"
Labels_Path = "D:/Projects/Machine Learning/anpr-rasp/data/my_model_SSD/label_map.pbtxt"
CNN_Model_Path = "D:/Projects/Machine Learning/anpr-rasp/data/my_model_CNN"
Detected_Plates_Path = "D:/Projects/Machine Learning/anpr-rasp/data/Detected_Plates/Croped Plates"
csv_filename = 'D:/Projects/Machine Learning/anpr-rasp/data/Detected_PlateszDetected_Plates.csv'
detection_threshold = 0.50
global camera_NO
camera_NO = 1

# ---------------------- Import Models ----------------------------------------------------------------------

model_SSD = tf.saved_model.load(SSD_Model_Path)
category_index = label_map_util.create_category_index_from_labelmap(Labels_Path , use_display_name=True)
model_CNN = keras.models.load_model(CNN_Model_Path , compile=False)

# ------------------------------------------------------------------------------------------------------

#real_time()
#model_show(model_SSD, '../ANPR/Model/Plates_Ex/2.jpg')

def switch_cam():
    global camera_NO
    stop()

    if camera_NO == 0:
        camera_NO = 1
    else:
        camera_NO = 0

    real_time()

main_interface=tkinter.Tk()
main_interface.title("Automatic Number Plate Detector")

frame=np.random.randint(0,255,[100,100,3],dtype='uint8')
#img = ImageTk.PhotoImage(Image.fromarray(frame))

paneeli_image=tkinter.Label(main_interface)
paneeli_image.grid(row=0,column=0,columnspan=10,pady=1,padx=10)

message=" "
paneeli_text=tkinter.Label(main_interface,text=message)
paneeli_text.grid(row=1,column=6,pady=1,padx=10)

global video_capture

def stop():
    global video_capture
    video_capture.release()
    cv2.destroyAllWindows()
    print("Stopped!")

btn_on=tkinter.Button(main_interface,text="Start",command=real_time,height=2)
btn_on.grid(row=1,column=0,columnspan=1,pady=10,padx=10)

btn_off=tkinter.Button(main_interface,text="Stop",command=stop,height=2)
btn_off.grid(row=1,column=1,columnspan=1,pady=10,padx=10)

btn_swtch=tkinter.Button(main_interface,text="Switch camera",command=switch_cam,height=2)
btn_swtch.grid(row=1,column=2,columnspan=1,pady=10,padx=10)

main_interface.mainloop()