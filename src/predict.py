from pathlib import Path
import os
import numpy as np
import tensorflow as tf
from moviepy.editor import *

# Adjust to use a different model/video file
VIDEO_FOLDER_PATH = 'E:\Projects\TwitchMontage\VideoCompilation\VideoFiles\\raw_clips'
MODEL_NAME = "valorant.h5"

ROOT_DIR = Path(__file__).parent.absolute().parent
MODEL_PATH = ROOT_DIR.joinpath(f"models/{MODEL_NAME}")\

DESIRED_VIDEO_LENGTH_IN_SECONDS = 720

def sort_folder():
    # Get list of all files only in the given directory
    list_of_files = filter( lambda x: os.path.isfile(os.path.join(VIDEO_FOLDER_PATH, x)),
                            os.listdir(VIDEO_FOLDER_PATH))
    # Sort list of files based on last modification time in ascending order
    list_of_files = sorted( list_of_files,
                            key = lambda x: os.path.getmtime(os.path.join(VIDEO_FOLDER_PATH, x)))
    return list_of_files
        
def rename_files(VIDEO_FOLDER_PATH):

    video_folder = sort_folder()
    for count, filename in enumerate(video_folder):
        dst = f"clip{str(count)}.mp4"
        src =f"{VIDEO_FOLDER_PATH}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{VIDEO_FOLDER_PATH}/{dst}"
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)

# selects valid clips and meets desired video length requirement
def run_selection(model_path=MODEL_PATH):
    rename_files(VIDEO_FOLDER_PATH)

    video_folder = os.listdir(VIDEO_FOLDER_PATH)
    results = []
    current_duration = 0
    video_idx = 0

    for video_file in video_folder:
        video_file_path = os.path.abspath(os.path.join(VIDEO_FOLDER_PATH, video_file))
        result = predict_ingame(video_file_path, model_path)
        video_idx += 1

        if(result[0] == True):
            results.append(result)
            current_duration += result[1]

        if(current_duration >= DESIRED_VIDEO_LENGTH_IN_SECONDS):
            break        
    
    return [video_idx, current_duration]

# returns prediction and video duration tuple
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
    duration = int(clip.duration)
    for t in range(1, duration):
        frame = clip.get_frame(t)
        resized_frame = tf.keras.preprocessing.image.smart_resize(frame, (224, 224), interpolation="nearest")
        frames.append(resized_frame)

    # predictions[0] = game, predictions[1] = nogame
    predictions = model.predict(np.array(frames))
    percentage = np.average(predictions, axis=0)
    print(f"Probability that video shows an ingame scene: {round(percentage[0] * 100, 4)}%")
    clip.close()
    
    prediction = round(percentage[0] * 100, 4) > 55

    return [prediction, duration]
    
selection = run_selection(MODEL_PATH)
print(selection)