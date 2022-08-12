from pathlib import Path
import os
import numpy as np
import tensorflow as tf
from moviepy.editor import *

# Adjust to use a different model/video file
VIDEO_FOLDER_PATH = 'E:\Projects\TwitchMontage\VideoCompilation\VideoFiles\\raw_clips'
MODEL_NAME = "valorant.h5"

ROOT_DIR = Path(__file__).parent.absolute().parent
MODEL_PATH = ROOT_DIR.joinpath(f"models/{MODEL_NAME}")

def run_predictions(model_path=MODEL_PATH):
    video_folder = os.listdir(VIDEO_FOLDER_PATH)
    predictions = []

    for video_file_name in video_folder:
        print
        video_file_path = os.path.abspath(os.path.join(VIDEO_FOLDER_PATH, video_file_name))
        predictions.append(predict_ingame(video_file_path, model_path))
        
    return predictions

def predict_ingame(VIDEO_PATH, MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except IOError:
        raise FileNotFoundError(f"No Model found at {MODEL_PATH}.")

    # VideoFileClip can only handle string as path (not PosixPath objects)
    try:
        clip = VideoFileClip(str(VIDEO_PATH))
    except IOError:
        raise FileNotFoundError(f"No Video found at {VIDEO_PATH}.")

    frames = []
    # Check one frame each second except for the first one (e.g. 9 checks for a 10s video)
    for t in range(1, int(clip.duration)):
        frame = clip.get_frame(t)
        resized_frame = tf.keras.preprocessing.image.smart_resize(frame, (224, 224), interpolation="nearest")
        frames.append(resized_frame)

    # predictions[0] = game, predictions[1] = nogame
    predictions = model.predict(np.array(frames))
    percentage = np.average(predictions, axis=0)
    print(f"Probability that video shows an ingame scene: {round(percentage[0] * 100, 4)}%")
    clip.close()
    
    return round(percentage[0] * 100, 4)
    

predictions = run_predictions(MODEL_PATH)
print(predictions)