# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 06:17:03 2021

@author: Ahmed Fayed
"""

from keras_facenet import FaceNet
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy.core.numeric import identity
from sklearn.preprocessing import Normalizer
from PIL import Image

from flask import Flask, render_template, request, send_from_directory, url_for, redirect, jsonify, json
import codecs
import base64

embedder = FaceNet()
l2_normalizer = Normalizer('l2')


def read_image(path):
    
    image = cv2.imread(path)
    
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image_RGB



def detect(image):
    
    detections = embedder.extract(image, threshold=0.95)
    
    return detections


def get_points(box):
    
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    
    x2 = x1 + width
    y2 = y1 + height
    
    return (x1, y1), (x2, y2)


database = dict()

def add_person(image, name):
    
    detections = detect(image)
    
    database[name] = detections[0]['embedding']
    



app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query_distance')
def query():
    return render_template('index2.html')


# Inpuet: one image
# Output: embedding for that image
@app.route('/encoding', methods=['POST'])
def encoding():
    img_upload = request.files['image']
    
    img = Image.open(img_upload)
    img_arr = np.array(img)

    detections = detect(img_arr)

    pt1, pt2 = get_points(detections[0]['box'])

    embedding = np.zeros((512))

    for res in detections:
        embedding = res['embedding']

    embedding_list = embedding.tolist()

   
    return jsonify({"embedding":embedding_list})

    
# Input: Two images
# Output: Identity
@app.route('/distance', methods=['POST'])
def distance():
    
    # data = request.json
    
    # embedding1 = np.array(data['embedding'])
    # embedding2 = np.array(data['embedding2'])
    # embedding1 = data['embedding']
    # embedding2 = data['embedding2']

    # embedding1 = np.fromstring(data['embedding'],dtype=float).reshape(512)
    # embedding2 = np.fromstring(data['embedding2'],dtype=float).reshape(512)


    # embedding_list = request.args.getlist('embedding_list')
    # embedding2_list = request.args.getlist('embedding2_list') 
 
    # embedding1 = np.array(embedding_list, dtype=float)
    # embedding2 = np.array(embedding2_list, dtype=float)


    img_upload = request.files['image']
    img2_upload = request.files['image2']
    

    img = Image.open(img_upload)
    img_arr = np.array(img)

    img2 = Image.open(img2_upload)
    img2_arr = np.array(img2)

    detections = detect(img_arr)
    detections2 = detect(img2_arr)


    embedding1 = np.zeros((512))
    embedding2 = np.zeros((512))


    for res in detections:
        embedding1 = res['embedding']
    
    for res in detections2:
        embedding2 = res['embedding']

    distance = embedder.compute_distance(embedding1, embedding2)
    distance = np.round(distance, decimals=1)
            

    identity = "different"

    if distance <= 0.3:
        identity = "same"



    return jsonify({"identity":identity})




if __name__ == '__main__':
    app.run(debug=True)




# ###############################################################################################################################
# ###############################################################################################################################

# ######      Detecting And Recognizing Multiple Faces in one Image    #########################################################


#################################################################################################################################
#################################################################################################################################



# # path = 'E:/Software/Experiments/IMG-20191208-WA0004.jpg'
# # image = read_image(path)


# # image_copy = image.copy()
# # image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# # detections = detect(image_copy)


# counter = 0

# for person_directory in os.listdir('E:/Software/Experiments/test/'):
#     person_dir = os.path.join('E:/Software/Experiments/test/', person_directory)
#     for image_name in os.listdir(person_dir):
#         image_path = os.path.join(person_dir, image_name)
#     image = read_image(image_path)
#     image_copy = image.copy()
#     image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    
#     # resizing experiment
#     # image_copy = cv2.resize(image_copy, (2000, 1800))
    
#     detections = detect(image_copy)
    
   

#     for res in detections:
        
#         pt1, pt2 = get_points(res['box'])
         
#         min_distance = 100
#         identity = "unknown"
#         for person in database:
            
#             embedding1 = database[person]
#             embedding2 = res['embedding']
#             distance = embedder.compute_distance(embedding1, embedding2)
#             distance = np.round(distance, decimals=1)
            
#             if distance < min_distance:
#                 min_distance = distance
#                 identity = person
                
                
        
#         if min_distance <= 0.3:
#             cv2.rectangle(image_copy, pt1, pt2, (0, 200, 200), 3)
#             cv2.putText(image_copy, identity + f'_{min_distance:.1f}', (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#         else:
#             cv2.rectangle(image_copy, pt1, pt2, (0, 0, 255), 3)
#             cv2.putText(image_copy, 'unknown', pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
#         print('identity:  ' + identity + ' ====>  ', min_distance)
     
         
#     image_name_2 = 'test' + '{}.jpg'    
#     cv2.imwrite(image_name_2.format(counter), image_copy)
#     counter += 1
#     # print(image.shape)
#     image_copy = cv2.resize(image_copy, (1000, 900))
#     # print(image_copy.shape)

#     plt.imshow(image_copy)
#     plt.show()
        
    
    
    

# # cv2.imwrite('IMG-20191208-WA0004.jpg', image_copy)
# # plt.imshow(image_copy)

# # print(image.shape)
# # image_copy = cv2.resize(image_copy, (1000, 900))
# # print(image_copy.shape)


# # while True:
    
# #     cv2.imshow('result', image_copy)
    
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
    
# # cv2.destroyAllWindows()

