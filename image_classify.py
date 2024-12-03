import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime

from torchvision.models import resnet50, ResNet50_Weights
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# Preprocess function for ResNet-50
preprocess_func = ResNet50_Weights.IMAGENET1K_V2.transforms()
categories = np.array(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])

# Cache the model loading
@st.cache_data
def load_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    return model

def make_prediction(model, processed_img):
    probs = model(processed_img.unsqueeze(0))
    probs = probs.softmax(1)
    probs = probs[0].detach().numpy()
    prob, idxs = probs[probs.argsort()[-5:][::-1]], probs.argsort()[-5:][::-1]
    return prob, idxs

def interpret_prediction(model, processed_img, target):
    interpretation_algo = IntegratedGradients(model)
    feature_imp = interpretation_algo.attribute(processed_img.unsqueeze(0), target=int(target))
    feature_imp = feature_imp[0].numpy()
    feature_imp = feature_imp.transpose(1, 2, 0)
    return feature_imp

# Define the sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Image Classifier", "Connect to Us"])

if page == "Image Classifier":
    ## Dashboard GUI
    st.title("ResNet-50 Image Classifier")
    upload = st.file_uploader(label="Upload Image:", type=["png", "jpg", "jpeg"])

    if upload or "img" in st.session_state:
        if upload:
            img = Image.open(upload)
            st.session_state.img = img
        else:
            img = st.session_state.img

        img_format = img.format.lower()  # Get image format (e.g., 'png', 'jpg')

        model = load_model()
        preprocessed_img = preprocess_func(img)
        probs, idxs = make_prediction(model, preprocessed_img)
        feature_imp = interpret_prediction(model, preprocessed_img, idxs[0])

        st.session_state.probs = probs
        st.session_state.idxs = idxs
        st.session_state.feature_imp = feature_imp

        main_fig = plt.figure(figsize=(12, 3))
        ax = main_fig.add_subplot(111)
        plt.barh(y=categories[idxs][::-1], width=probs[::-1], color=["dodgerblue"]*4 + ["tomato"])
        plt.title("Top 5 Probabilities", loc="center", fontsize=15)
        st.pyplot(main_fig, use_container_width=True)

        # Show integrated gradients attributions
        interp_fig, ax = viz.visualize_image_attr(
            feature_imp,
            sign="all",
            show_colorbar=True,
            title="Integrated Gradients Attributions",
            fig_size=(6, 6)
        )

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            main_fig = plt.figure(figsize=(6, 6))
            ax = main_fig.add_subplot(111)
            plt.imshow(img)
            plt.xticks([], [])
            plt.yticks([], [])
            st.pyplot(main_fig, use_container_width=True)

        with col2:
            st.pyplot(interp_fig, use_container_width=True)

        # Rename and save the image
        top_category = categories[idxs][0]
        new_filename = f"{top_category}.{img_format}"
        st.session_state.new_filename = new_filename

        # Save image to in-memory file
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img_format.upper())
        img_byte_arr = img_byte_arr.getvalue()
        st.session_state.img_byte_arr = img_byte_arr

        # Display the download button
        st.download_button(label="Download Renamed Image", data=img_byte_arr, file_name=new_filename, mime=f"image/{img_format}")

elif page == "Connect to Us":
    st.title("Connect to Us")
    st.write("We'd love to hear from you!")

    # Feedback section
    st.subheader("Feedback")
    st.write("Did you find the output useful?")
    col1, col2 = st.columns([1, 1])

    if "new_filename" in st.session_state:
        with col1:
            if st.button("👍 Like"):
                with open("feedback.txt", "a") as f:
                    f.write(f"Like for {st.session_state.new_filename}\n")
                st.success("Thank you for your feedback!")

        with col2:
            if st.button("👎 Dislike"):
                with open("feedback.txt", "a") as f:
                    f.write(f"Dislike for {st.session_state.new_filename}\n")
                st.success("Thank you for your feedback!")

    # User comment section
    comment = st.text_area("Leave your comment here:")
    if st.button("Submit Comment"):
        with open("user_comments.txt", "a") as f:
            f.write(f"{datetime.now()}: {comment}\n")
        st.success("Thank you for your comment!")
