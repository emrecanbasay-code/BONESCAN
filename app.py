from typing import Any
import singleinference_yolov7
from singleinference_yolov7 import SingleInference_YOLOV7
import os
import streamlit as st
import logging
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import base64
from streamlit_paste_button import paste_image_button as pbutton

st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="centered",
)


class Streamlit_YOLOV7(SingleInference_YOLOV7):
    '''
    Streamlit app for bone fracture detection in X-ray images
    '''
    

    def __init__(self,):
        self.logging_main=logging
        self.logging_main.basicConfig(level=self.logging_main.DEBUG)

    

    def new_yolo_model(self,img_size,path_yolov7_weights,path_img_i,device_i='cpu'):
        '''
        
        INPUTS:
        img_size,                 #int#   #this is the yolov7 model size, should be square so 640 for a square 640x640 model etc.
        path_yolov7_weights,      #str#   #this is the path to your yolov7 weights 
        path_img_i,               #str#   #path to a single .jpg image for inference (NOT REQUIRED, can load cv2matrix with self.load_cv2mat())

        OUTPUT:
        predicted_bboxes_PascalVOC   #list#  #list of values for detections containing the following (name,x0,y0,x1,y1,score)

        CREDIT
        Please see https://github.com/WongKinYiu/yolov7.git for Yolov7 resources (i.e. utils/models)
        @article{wang2022yolov7,
            title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
            author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
            journal={arXiv preprint arXiv:2207.02696},
            year={2022}
            }
        
        '''
        super().__init__(img_size,path_yolov7_weights,path_img_i,device_i=device_i)
    def main(self):
        st.markdown(
            """
            <style>
            .stMainBlockContainer h1, .stMainBlockContainer h2 {
                text-align: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.title('🦴 Bone Fracture Detection')
        st.caption('Upload an X-ray image to detect fractures')

        # --- Sidebar ---
        with st.sidebar:
            st.write("**Bone Fracture Detection**")
            st.caption("Appendicular X-ray images")
            st.divider()
            self.conf_selection=st.select_slider('Confidence Threshold',options=[0.05,0.1,0.15,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95])
            st.divider()
            st.caption("⚠️ Not for medical diagnosis")

        # --- Default image ---
        self.response=requests.get(self.path_img_i)
        self.img_screen=Image.open(BytesIO(self.response.content))
        st.image(self.img_screen, caption=self.capt, use_container_width=True, channels="RGB", output_format="auto")

        # --- Upload + Predict ---
        self.im0=np.array(self.img_screen.convert('RGB'))
        self.load_image_st()

        st.write("")
        predictions = st.button('🔍 Predict', type='primary')
        if predictions:
            self.predict()
            predictions=False

    def load_image_st(self):

        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_img=st.file_uploader('Upload an image')
        with col2:
            st.write("")  # spacing to align with file uploader
            paste_result = pbutton(
                label="📋 Paste from Clipboard",
                text_color="#ffffff",
                background_color="#3498db",
                hover_background_color="#2980b9",
                key="paste_btn",
                errors="raise",
            )

        # Check for pasted image from clipboard
        if paste_result.image_data is not None:
            st.image(paste_result.image_data, caption='📋 Pasted from clipboard')
            self.im0 = np.array(paste_result.image_data.convert('RGB'))
            return self.im0

        if uploaded_img is not None:
            self.img_data=uploaded_img.getvalue()
            st.image(self.img_data)
            self.im0=Image.open(BytesIO(self.img_data))#.convert('RGB')
            self.im0=np.array(self.im0)

            return self.im0
        elif self.im0 is not None:
            return self.im0
        else:
            return None
    
    def predict(self):
        self.conf_thres=self.conf_selection
        
        with st.spinner('Loading image...'):
            self.load_cv2mat(self.im0)
        
        with st.spinner('Making inference...'):
            self.inference()

        self.img_screen=Image.fromarray(self.image).convert('RGB')
        
        self.capt='DETECTED:'
        if len(self.predicted_bboxes_PascalVOC)>0:
            for item in self.predicted_bboxes_PascalVOC:
                name=str(item[0])
                conf=str(round(100*item[-1],2))
                self.capt=self.capt+ ' name='+name+' confidence='+conf+'%, '
        st.image(self.img_screen, caption=self.capt, use_container_width=True, channels="RGB", output_format="auto")
        self.image=None
    

if __name__=='__main__':
    app=Streamlit_YOLOV7()

    #INPUTS for YOLOV7
    img_size=2560
    path_yolov7_weights="weights/best.pt"
    path_img_i="https://cdn.prod.website-files.com/66d9c5d9b2cb9d538db62b7d/67bf872bc901484636782f79_AI%20in%20Fracture.jpg"
    #INPUTS for webapp
    app.capt="MD:ECB "
    app.new_yolo_model(img_size,path_yolov7_weights,path_img_i)
    app.conf_thres=0.05
    
    app.load_model() #Load the yolov7 model
    
    app.main()
