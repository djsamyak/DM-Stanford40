import os
import cv2
import shutil
import statistics
import numpy as np
import streamlit as st 
from pytube import YouTube
import matplotlib.pyplot as plt
from skimage.transform  import resize
from models.hybrid import model as hybrid
from models.resnet50 import model as res50 
from models.resnet101 import model as res101
from models.inceptionv3 import model as incepV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

class_label_dict = {
        0:'applauding',                        
        1:'blowing_bubbles',                        
        2:'brushing_teeth',                      
        3:'cleaning_the_floor',                
        4:'climbing',                        
        5:'cooking',                                
        6:'cutting_trees',                        
        7:'cutting_vegetables',                
        8:'drinking',                        
        9:'feeding_a_horse',                        
        10:'fishing',                                
        11:'fixing_a_bike',                        
        12:'fixing_a_car', 
        13:'gardening',                        
        14:'holding_an_umbrella',                
        15:'jumping',                                
        16:'looking_through_a_microscope',        
        17:'looking_through_a_telescope',        
        18:'playing_guitar',                        
        19:'playing_violin',                        
        20:'pouring_liquid',                        
        21:'pushing_a_cart',                        
        22:'reading',                                
        23:'phoning',                                
        24:'riding_a_bike',                        
        25:'riding_a_horse', 
        26:'rowing_a_boat',                        
        27:'running',                                
        28:'shooting_an_arrow',                
        29:'smoking',                                
        30:'taking_photos',                        
        31:'texting_message',                        
        32:'throwing_frisby',                        
        33:'using_a_computer',                
        34:'walking_the_dog',                        
        35:'washing_dishes',                        
        36:'watching_TV',                        
        37:'waving_hands',                        
        38:'writing_on_a_board',                
        39:'writing_on_a_book'                                              
    }

PATH_InceptionResNetV2 = r'./weights/incepResV2_dropout2_lr4_schd.h5'
PATH_InceptionV3 = r'./weights/InceptionV3.h5'
PATH_ResNet101 = r'./weights/resnet101_reg.h5'
PATH_ResNet50 = r'./weights/resnet50_4_12.h5'

st.title("Human Activity Recognition")
st.sidebar.title("Select Model")
selected_model = st.sidebar.selectbox('Model', ['---select---','ResNet50','ResNet101','InceptionV3','InceptionResNetV2'])

if selected_model == 'ResNet50':
    current_model = res50
    current_path = PATH_ResNet50
elif selected_model == 'ResNet101':
    current_model = res101
    current_path = PATH_ResNet101
elif selected_model == 'InceptionV3':
    current_model = incepV3
    current_path = PATH_InceptionV3
elif selected_model == 'InceptionResNetV2':
    current_model = hybrid
    current_path = PATH_InceptionResNetV2

operation_mode = st.radio('Operation Mode:',['Image','Video'])
uploaded_img = None
entered_url = None

if operation_mode == 'Image':
    st.write("Upload an Image to analyze.")
    uploaded_img = st.file_uploader('File uploader',type=['jpg','png'])
    
    if uploaded_img != None:
        st.write("Selected Image:")
        st.image(uploaded_img,width = 400)

        if selected_model == '---select---':
            st.warning("Model has not been selected")
        else:

            img = image.load_img(uploaded_img, target_size=(256, 256))
            img = image.img_to_array(img)
            final_image = np.expand_dims(img, axis=0)
            
            if selected_model == 'InceptionResNetV2':
                final_image = preprocess_input(final_image)

            if st.button("Predict"):
                model = current_model(current_path)
                yhat = list(model.predict(final_image))
                idx = np.argmax(yhat)
                st.write(f"Predicted Class: {class_label_dict[idx]}")
                st.write(f"Confidence Value: {yhat[0][idx]*100:.2f}%")
        
elif operation_mode == 'Video':

    url_placeholder = ' '
    entered_url =  st.text_input('Enter a youtube video link to analyze: ',value=' ')

    if entered_url != url_placeholder:
        if selected_model == '---select---':
            st.warning("Model has not been selected")
        else:
            with st.spinner('Processing Video'):
                link = entered_url
                yt = YouTube(link)
                out_file = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download('sample')
                os.rename(out_file, './sample/sample.mp4')
                cap = cv2.VideoCapture('./sample/sample.mp4')
                n=0
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    cv2.imwrite('./sample/'+str(n)+'.jpg',frame)
                    n+=1

                cap.release()
            st.info('Video Processed!')
            
            index = []
            model = current_model(current_path)

            video_file = open('./sample/sample.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            video_file.close()

            if st.button("Predict"):
                for i in range(0,2):
                    img = image.load_img('./sample/' + str(i) + '.jpg', target_size=(256, 256))

                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)

                    if selected_model == 'InceptionResNetV2':
                        x = preprocess_input(x)

                    pred = model.predict(x)
                    pred = pred.tolist()
                    index.append(pred[0].index(max(pred[0])))

                idx = statistics.mode(index)
                st.write("Prediction: ", class_label_dict[idx])
                
            shutil.rmtree(".\sample")