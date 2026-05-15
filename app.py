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
from streamlit_paste_button import paste_image_button as pbutton

st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Medical Dashboard CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --medical-blue: #004a99;
        --medical-blue-hover: #003366;
        --bg-color: #f8f9fa;
        --card-bg: #ffffff;
        --text-primary: #212529;
        --text-secondary: #6c757d;
        --success: #28a745;
        --warning: #ffc107;
        --danger: #dc3545;
        --border-radius: 12px;
        --shadow: 0 2px 12px rgba(0,0,0,0.08);
    }

    html {
        font-family: 'Inter', sans-serif;
    }

    /* Main container styling */
    .main .block-container {
        background-color: var(--bg-color);
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 2rem 0;
        margin-bottom: 1rem;
    }

    .main-header h1 {
        color: var(--medical-blue);
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 400;
    }

    /* Card styling */
    .card {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .input-card {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        padding: 2rem;
        text-align: center;
        border: 2px dashed #dee2e6;
        transition: all 0.3s ease;
    }

    .input-card:hover {
        border-color: var(--medical-blue);
    }

    /* Button styling */
    .stButton > button {
        background-color: var(--medical-blue) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background-color: var(--medical-blue-hover) !important;
        box-shadow: 0 4px 12px rgba(0,74,153,0.3) !important;
    }

    /* File uploader styling */
    .uploadedFile {
        border-radius: 8px;
    }

    /* Image container */
    .image-container {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        padding: 1rem;
        text-align: center;
    }

    .image-container img {
        border-radius: 8px;
        max-width: 100%;
    }

    .image-container p {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    /* Metrics container */
    .metrics-container {
        display: flex;
        justify-content: space-around;
        gap: 1rem;
        margin-top: 1rem;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--card-bg) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--card-bg);
    }

    .css-1d391kg .e1nzilvr1 {
        background-color: var(--bg-color);
    }

    /* Signature styling */
    .signature {
        position: fixed;
        bottom: 1rem;
        right: 1.5rem;
        color: var(--text-secondary);
        font-size: 0.75rem;
        font-weight: 400;
        opacity: 0.7;
        z-index: 999;
        pointer-events: none;
    }

    /* Status indicators */
    .status-detected {
        color: var(--success);
        font-weight: 600;
    }

    .status-clear {
        color: var(--success);
        font-weight: 600;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>

    <div class="signature">By Md. ECB</div>
    """,
    unsafe_allow_html=True,
)


class Streamlit_YOLOV7(SingleInference_YOLOV7):
    '''
    Streamlit app for bone fracture detection in X-ray images
    '''

    def __init__(self):
        self.logging_main = logging
        self.logging_main.basicConfig(level=self.logging_main.DEBUG)

    def new_yolo_model(self, img_size, path_yolov7_weights, path_img_i, device_i='cpu'):
        '''
        Initialize YOLOv7 model for inference.

        INPUTS:
        img_size: int - YOLOv7 model size (square, e.g., 640 for 640x640)
        path_yolov7_weights: str - Path to YOLOv7 weights file
        path_img_i: str - Path to default image (not used in new design)
        device_i: str - Device to run inference on ('cpu' or 'cuda')

        OUTPUT:
        predicted_bboxes_PascalVOC: list - Detection results with format
                                         (name, x0, y0, x1, y1, score)

        CREDIT:
        YOLOv7: https://github.com/WongKinYiu/yolov7
        @article{wang2022yolov7,
            title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
            author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
            journal={arXiv preprint arXiv:2207.02696},
            year={2022}
        }
        '''
        super().__init__(img_size, path_yolov7_weights, path_img_i, device_i=device_i)

    def init_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'image_uploaded' not in st.session_state:
            st.session_state.image_uploaded = False
        if 'prediction_made' not in st.session_state:
            st.session_state.prediction_made = False
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None

    def render_header(self):
        """Render application header."""
        st.markdown(
            """
            <div class="main-header">
                <h1>🦴 Bone Fracture Detection</h1>
                <p>AI-Powered X-Ray Analysis for Medical Professionals</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    def render_sidebar(self):
        """Render sidebar with settings."""
        with st.sidebar:
            st.markdown("### ⚙️ Settings")
            st.divider()

            st.write("**Confidence Threshold**")
            self.conf_selection = st.select_slider(
                'Detection Sensitivity',
                options=[0.05, 0.1, 0.15, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                value=0.05,
                format_func=lambda x: f"{x:.2f}"
            )

            st.divider()

            st.markdown("### 📊 Model Info")
            st.info("""
            **Model:** YOLOv7
            **Input Size:** 2560x2560
            **Classes:** Fracture Detection

            Trained on appendicular
            X-ray images.
            """)

            st.divider()
            st.caption("⚠️ **Not for medical diagnosis**")
            st.caption("This tool is for research and educational purposes only.")

    def get_input_panel(self):
        """Render input panel for file upload and paste functionality."""
        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_img = st.file_uploader(
                '📁 Upload X-Ray Image',
                type=['jpg', 'jpeg', 'png'],
                label_visibility="collapsed"
            )

        with col2:
            st.write("")
            paste_result = pbutton(
                label="📋 Paste",
                text_color="#ffffff",
                background_color="#004a99",
                hover_background_color="#003366",
                key="paste_btn",
                errors="raise",
            )

        # Handle pasted image
        if paste_result.image_data is not None:
            st.session_state.current_image = np.array(paste_result.image_data.convert('RGB'))
            st.session_state.image_uploaded = True
            st.session_state.prediction_made = False
            return st.session_state.current_image

        # Handle uploaded image
        if uploaded_img is not None:
            img_data = uploaded_img.getvalue()
            img = Image.open(BytesIO(img_data))
            st.session_state.current_image = np.array(img.convert('RGB'))
            st.session_state.image_uploaded = True
            st.session_state.prediction_made = False
            return st.session_state.current_image

        return None

    def render_empty_state(self):
        """Render empty state when no image is uploaded."""
        st.markdown(
            """
            <div class="input-card">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
                <h3 style="color: var(--text-primary); margin-bottom: 0.5rem;">
                    Upload an X-Ray Image
                </h3>
                <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
                    Supported formats: JPG, PNG, JPEG
                </p>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">
                    Or paste an image directly from your clipboard
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info("💡 **Tip:** Use `Windows + Shift + S` to take a screenshot on Windows, then paste here.")

    def render_image_preview(self):
        """Render preview of uploaded image."""
        if st.session_state.current_image is not None:
            st.markdown("### 📸 Uploaded Image")
            st.image(
                st.session_state.current_image,
                use_container_width=True,
                output_format="auto"
            )

    def run_prediction(self):
        """Run YOLOv7 inference on the uploaded image."""
        self.conf_thres = self.conf_selection

        with st.spinner('🔄 Processing X-ray image...'):
            self.load_cv2mat(st.session_state.current_image)

        with st.spinner('🤖 AI Analyzing...'):
            self.inference()

        # Store predictions in session state
        st.session_state.predictions = self.predicted_bboxes_PascalVOC
        st.session_state.prediction_made = True
        st.session_state.result_image = Image.fromarray(self.image).convert('RGB')

    def display_results(self):
        """Display detection results in dashboard format."""
        if not st.session_state.prediction_made:
            return

        predictions = st.session_state.predictions

        # Calculate metrics
        fracture_count = len(predictions)
        avg_confidence = np.mean([p[-1] for p in predictions]) * 100 if predictions else 0
        max_confidence = np.max([p[-1] for p in predictions]) * 100 if predictions else 0

        # Status based on detection
        if fracture_count > 0:
            status = "Fractures Detected"
            status_color = "🔴"
        else:
            status = "No Fractures Detected"
            status_color = "🟢"

        # Display metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="Fractures Found", value=fracture_count, delta="Detected" if fracture_count > 0 else "Clear")

        with col2:
            st.metric(label="Avg. Confidence", value=f"{avg_confidence:.1f}%")

        with col3:
            st.metric(label="Max Confidence", value=f"{max_confidence:.1f}%")

        with col4:
            st.metric(label="Status", value=status, delta=status_color if fracture_count > 0 else "✓ Clear")

        # Side-by-side images
        st.markdown("### 🔍 Comparison View")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Original X-Ray")
            st.image(
                st.session_state.current_image,
                use_container_width=True,
                output_format="auto"
            )

        with col2:
            st.markdown("#### AI Detection")
            st.image(
                st.session_state.result_image,
                use_container_width=True,
                output_format="auto"
            )

        # Detailed results expander
        if predictions:
            with st.expander("📋 Detailed Detection Results"):
                # Create data for table
                table_data = []
                for i, pred in enumerate(predictions, 1):
                    table_data.append({
                        "#": i,
                        "Class": pred[0],
                        "Confidence": f"{pred[-1] * 100:.2f}%",
                        "Bounding Box": f"({pred[1]:.0f}, {pred[2]:.0f}) → ({pred[3]:.0f}, {pred[4]:.0f})"
                    })

                # Display as table
                st.table(table_data)
        else:
            st.success("✅ No fractures detected in the uploaded X-ray image.")

    def main(self):
        """Main application flow."""
        # Initialize session state
        self.init_session_state()

        # Render header
        self.render_header()

        # Render sidebar
        self.render_sidebar()

        # Input section
        st.markdown("### 📤 Image Input")

        image = self.get_input_panel()

        # Show content based on state
        if not st.session_state.image_uploaded:
            self.render_empty_state()
        else:
            # Show predict button
            st.markdown("---")
            predict_col = st.columns([1, 2, 1])[1]
            with predict_col:
                if st.button('🔍 Run AI Analysis', type='primary', use_container_width=True):
                    self.run_prediction()

            # Show results if prediction made
            if st.session_state.prediction_made:
                st.markdown("---")
                self.display_results()
            else:
                # Just show image preview before prediction
                self.render_image_preview()


if __name__ == '__main__':
    app = Streamlit_YOLOV7()

    # INPUTS for YOLOv7
    img_size = 2560
    path_yolov7_weights = "weights/best.pt"
    path_img_i = ""  # Empty - no default image

    # Initialize model
    app.new_yolo_model(img_size, path_yolov7_weights, path_img_i)
    app.conf_thres = 0.05

    # Load the YOLOv7 model
    app.load_model()

    # Run the app
    app.main()
