import numpy as np
import gradio as gr
from PIL import Image
import tensorflow as tf
import keras
import cv2


def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))

def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)

def preprocess_image(image):
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
        img = lightEnhance_model(img, training=False)
    if "Super Resolution" in select_service:
        img = superRes_model(img, training=False)
    if "Denoising" in select_service:
        img = denoise_model(img, training=False)
    return img




#Interface fuction
def img_infer(select_service, input_img):
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
    FRAME_BATCH = 2

    original_vid = cv2.VideoCapture(input_vid)
    frame_count = original_vid.get(cv2.CAP_PROP_FRAME_COUNT)
    enhance_vid = initialize_output_vid(original_vid, output_name='enhance_vid.mp4')

    batch_frame = []
    control = -1
    for num in range(int(frame_count)):
        control = (control+1) % FRAME_BATCH
        #prepare batch frame
        ret, frame = original_vid.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_frame.append(preprocess_image(frame))
        #enhance frame on batch
        if control==(FRAME_BATCH-1) or num==int(frame_count)-1:
            enhance_batch_frame = infer(np.vstack(batch_frame), select_service)
            for idx, enhance_frame in enumerate(enhance_batch_frame):
                enhance_frame = postprocess_image(enhance_frame, type="frame")
                enhance_vid.write(cv2.cvtColor(enhance_frame, cv2.COLOR_RGB2BGR))
                print(str(num+idx-FRAME_BATCH+2) + '/' + str(frame_count) + ' Complete!')
                print()
                batch_frame = []
    enhance_vid.release()
    original_vid.release()
    cv2.destroyAllWindows()
    return f"enhance_vid.mp4"



example_imgs = [[["Light Enhance"],'./example_imgs/lightEnhance_example1.png'], 
                [["Super Resolution"], './example_imgs/superRes_example1.png'],
                [["Denoising"], './example_imgs/denoing_example1.png']
]

example_vids = [[["Super Resolution", "Denoising"], './example_vids/superRes_example1.mp4']]

lightEnhance_model = tf.saved_model.load('./lightEnhance_11_8_2')
superRes_model = tf.saved_model.load('./superRes_11_8_1')
denoise_model = tf.saved_model.load('./denoise_11_8_1')


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

demo.launch(share=False)






