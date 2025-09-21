# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Lip2Text')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('Lip2Text') 

# 🧠 Load model once
model = load_model()

# ----------------------------------------
# 🎬 Section 1: Prepare and Process Raw Video
# ----------------------------------------
st.header("🎬 Prepare and Process Raw Video")

raw_file = st.file_uploader("Upload a raw video (.mp4)", type=["mp4"], key="raw_upload")
if raw_file is not None:
    raw_name = os.path.splitext(raw_file.name)[0]
    raw_path = f"{raw_name}_raw.mp4"
    trimmed_path = f"{raw_name}_trimmed.mp4"
    resized_path = f"{raw_name}_resized.mp4"
    final_mpg_path = os.path.join("..", "data", "s1", f"{raw_name}.mpg")

    # Save raw upload
    with open(raw_path, "wb") as f:
        f.write(raw_file.getbuffer())
    st.success("Raw video uploaded.")

    # Step 1: Trim to 3 seconds
    os.system(f"ffmpeg -i {raw_path} -t 3 -c copy {trimmed_path} -y")
    st.info("Trimmed to 3 seconds.")

    # Step 2: Resize to 360x288
    os.system(f"ffmpeg -i {trimmed_path} -vf scale=360:288 {resized_path} -y")
    st.info("Resized to 360×288.")

    # Step 3: Convert to .mpg
    os.system(f"ffmpeg -i {resized_path} -qscale:v 2 {final_mpg_path} -y")
    st.success(f"Converted to .mpg and saved to {final_mpg_path}")

    # Step 4: Preview processed video
    os.system(f"ffmpeg -i {final_mpg_path} -vcodec libx264 test_video.mp4 -y")
    st.video(open("test_video.mp4", "rb").read())

    # Step 5: Run inference
    st.info("Running LipNet prediction...")
    video, annotations = load_data(tf.convert_to_tensor(final_mpg_path))
    video = [tf.squeeze(frame).numpy().astype('uint8') for frame in video]
    imageio.mimsave('animation.gif', video, duration=100)
    st.image('animation.gif', width=400)

    yhat = model.predict(tf.expand_dims(video, axis=0))
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    st.text(f"Raw tokens: {decoder}")
    converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
    st.success(f"Predicted text: {converted_prediction}")

# ----------------------------------------
# 📤 Section 2: Upload Preprocessed .mpg Video
# ----------------------------------------
st.header("📤 Upload Preprocessed Video (.mpg)")

uploaded_file = st.file_uploader("Upload a silent video (.mpg)", type=["mpg"], key="mpg_upload")
if uploaded_file is not None:
    upload_path = os.path.join("..", "data", "s1", uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded and saved to {upload_path}")

# 📁 List available videos including newly uploaded ones
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# 🧪 Inference block
if selected_video:
    file_path = os.path.join('..', 'data', 's1', selected_video)

    # 🎥 Convert to mp4 for display
    os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

    col1, col2 = st.columns(2)

    with col1:
        st.info('The video below displays the converted video in mp4 format')
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        video = [tf.squeeze(frame).numpy().astype('uint8') for frame in video]
        imageio.mimsave('animation.gif', video, duration=100)
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(f"Raw tokens: {decoder}")

        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.success(f"Predicted text: {converted_prediction}")
