import custom_modules
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import gradio as gr

AUTOTUNE = tf.data.AUTOTUNE


def preprocess_image(image):
    image = image.resize(((image.width//4)*4, (image.height//4)*4))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_image(model_output, type):
    tf.experimental.numpy.experimental_enable_numpy_behavior(prefer_float32=True)
    model_output = model_output * 255.0
    model_output = model_output.clip(0, 255)
    if type == "img":
        image = model_output[0].reshape(
            (np.shape(model_output)[1], np.shape(model_output)[2], 3)
        )
        image = Image.fromarray(np.uint8(image))
    elif type == "frame":
        image = model_output.reshape(
            (np.shape(model_output)[0], np.shape(model_output)[1], 3)
        )
        image = np.uint8(image)
    return image

def infer(img, select_service):
    if "Light Enhance" in select_service:
        img = lightEnhance_model.predict(img, batch_size=AUTOTUNE)
    if "Denoising" in select_service:
        img = denoise_model.predict(img, batch_size=AUTOTUNE)
    if "Super Resolution" in select_service:
        img = superRes_model.predict(img, batch_size=AUTOTUNE)
    return img




#Interface fuction
def img_infer(select_service, input_img):
    input_img = input_img.convert('RGB')
    preprocessed_img = preprocess_image(input_img)
    model_output = infer(preprocessed_img, select_service)
    post_processed_image = postprocess_image(model_output, type="img")
    return post_processed_image


def initialize_output_vid(original_vid, output_name):
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  fps =  original_vid.get(cv2.CAP_PROP_FPS)
  width = original_vid.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = original_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
  return cv2.VideoWriter(output_name, fourcc, fps, (int(width), int(height)))

def vid_infer(select_service, input_vid):
    original_vid = cv2.VideoCapture(input_vid)
    frame_count = original_vid.get(cv2.CAP_PROP_FRAME_COUNT)
    enhance_vid = initialize_output_vid(original_vid, output_name='enhance_vid.mp4')

    batch_frame = []
    control = -1
    for num in range(int(frame_count)):
        #prepare batch frame
        ret, frame = original_vid.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_frame.append(preprocess_image(frame))
        #enhance frame on batch
        enhance_batch_frame = infer(np.vstack(batch_frame), select_service)
        for idx, enhance_frame in enumerate(enhance_batch_frame):
            enhance_frame = postprocess_image(enhance_frame, type="frame")
            enhance_vid.write(cv2.cvtColor(enhance_frame, cv2.COLOR_RGB2BGR))
            print(str(idx) + '/' + str(frame_count) + ' Complete!')
            print()
            batch_frame = []
            
    enhance_vid.release()
    original_vid.release()
    cv2.destroyAllWindows()
    return f"enhance_vid.mp4"



example_imgs = [[["Light Enhance"],r'/example_imgs/lightEnhance_example1.png'], 
                [["Super Resolution"], r'/example_imgs/superRes_example1.png'],
                [["Denoising"], r'/example_imgs/denoing_example1.png']
]

example_vids = [[["Super Resolution", "Denoising"], r'/example_vids/superRes_example1.mp4']]

lightEnhance_model = tf.keras.models.load_model(r"lightEnhance_testModel.keras", compile=False)
superRes_model = tf.keras.models.load_model(r"superRes_testModel.keras", compile=False)
denoise_model = tf.keras.models.load_model(r"denoising_testModel.keras", compile=False)


img_iface = gr.Interface(
    title="Image Enhancement",
    fn=img_infer,
    inputs=[
        gr.CheckboxGroup(["Light Enhance", "Super Resolution", "Denoising"], value=["Light Enhance", "Super Resolution", "Denoising"], label="Select Service"),
        gr.Image(label="image", type="pil")
    ],
    outputs="image",
    examples=example_imgs,
    cache_examples=True
)


vid_iface = gr.Interface(
    title="Video Enhancement",
    fn=vid_infer,
    inputs=[
        gr.CheckboxGroup(["Light Enhance", "Super Resolution", "Denoising"], value=["Light Enhance", "Super Resolution", "Denoising"], label="Select Service"),
        gr.Video(label="video")
    ],
    outputs="video",
    examples=example_vids,
    cache_examples=True
)

demo = gr.TabbedInterface([img_iface, vid_iface], ["Image Enhancement", "Video Enhancement"])

demo.launch(share=True)