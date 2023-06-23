# import gradio as gr
# import os


# def video_identity(video):
#     return video


# demo = gr.Interface(video_identity, 
#                     gr.Video(), 
#                     "playable_video", 
#                     cache_examples=True)

# if __name__ == "__main__":
#     demo.launch()








import gradio as gr
import cv2
import numpy as np

def generate_video():
    # Define the video dimensions and frame rate
    frame_width = 640
    frame_height = 480
    fps = 30

    # Create a VideoWriter object to save the output video
    output_video = cv2.VideoWriter("output_video.avi",
                                    cv2.VideoWriter_fourcc(*"MJPG"),
                                    fps,
                                    (frame_width, frame_height))

    # Generate the video frames
    for i in range(300):
        # Create a blank frame
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        # Add some text to the frame
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Write the frame to the output video
        output_video.write(frame)

    # Release the VideoWriter object
    output_video.release()

    # Return the output video file as a bytes object
    with open("output_video.avi", "rb") as f:
        video_bytes = f.read()
    return video_bytes

iface = gr.Interface(
    fn=generate_video,
    inputs=None,
    outputs="video",
    title="Video Generator",
    description="Generates a video with text overlays.",
    allow_flagging=False,
    live=False,
    theme="default"
)

iface.launch()