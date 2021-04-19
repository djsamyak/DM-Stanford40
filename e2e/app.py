import numpy as np
import streamlit as st 
import matplotlib.pyplot as plt
from skimage.transform  import resize
from models.hybrid import model as hybrid

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
        49:'writing_on_a_book'                                              
    }

PATH_InceptionResNetV2 = r'../../incepResV2_dropout2_lr4_schd.h5'
PATH_InceptionV3 = r''
PATH_ResNet101 = r''
PATH_ResNet50 = r''

st.sidebar.title("Select Model:")
selected_model = st.sidebar.selectbox('Model', ['ResNet50','ResNet101','InceptionV3','InceptionResNetV2'])

if selected_model == 'ResNet50':
    pass
elif selected_model == 'ResNet101':
    pass
elif selected_model == 'InceptionV3':
    pass
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

        image = plt.imread(uploaded_img)
        image_resized = resize(image, (256,256), anti_aliasing=True)
        image = np.expand_dims(image_resized, axis=0)

        if st.button("Predict"):
            model = current_model(current_path)
            yhat = list(model.predict(image))
            idx = yhat.index(max(yhat))
            st.write(f"Predicted Class: {class_label_dict[idx]}")

elif operation_mode == 'Video':
    entered_url = st.text_area('Enter a YouTube URL to analyze')