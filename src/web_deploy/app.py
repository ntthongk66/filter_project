import gradio as gr
from filter_vid_image import filter_vid_image

def filter_image(image):
    output = filter_vid_image(True, image)
    return output

def filter_vid(vid):
    output = filter_vid_image(False, vid)
    return output
    
# app = gr.Interface(inputs='image', fn=filter_image, outputs="image")

# app.launch()


app_vid = gr.Interface(filter_vid, gr.Video(), "playable_video")