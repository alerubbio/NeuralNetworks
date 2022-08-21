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
    clip_duration = int(clip.duration)
    for t in range(1, clip_duration):
        frame = clip.get_frame(t)
        resized_frame = tf.keras.preprocessing.image.smart_resize(frame, (224, 224), interpolation="nearest")
        frames.append(resized_frame)

    # predictions[0] = game, predictions[1] = nogame
    predictions = model.predict(np.array(frames))
    percentage = np.average(predictions, axis=0)
    print(f"Probability that video shows an ingame scene: {round(percentage[0] * 100, 4)}% " + "\n" + str(VIDEO_PATH))
    clip.close()
    
    # boolean
    prediction = round(percentage[0] * 100, 4) >= 70

    return [prediction, clip_duration]
    
# selects valid clips and meets desired video length requirement
def run_selection(model_path=MODEL_PATH):
    video_folder = os.listdir(VIDEO_FOLDER_PATH)
    results = []
    current_duration = 0
    video_idx = 0

    for video_file in video_folder:
        video_file_path = os.path.abspath(os.path.join(VIDEO_FOLDER_PATH, video_file))
        result = [video_file, predict_ingame(video_file_path, model_path)]
        video_idx += 1

        if(result[1][0] == True):
            results.append(result)
            current_duration += result[1][1]

        if(current_duration >= DESIRED_VIDEO_LENGTH_IN_SECONDS):
            break        
    
    return [results, video_idx, current_duration]

def write_results2file(results):

    with open('VideoCompilation/ClipData/valid_clips.txt', 'w') as fp:
        for result in results:
            # write each item on a new line
            fp.write("%s\n" % result[0])


selection = run_selection(MODEL_PATH)

write_results2file(selection[0])



print('---------------------------------------------------------------------------------\n')
print('Selection Data: \n')
print(f'Clip count: {selection[1]} \nCompilation Duration: {selection[2]}s')


# for result in selection[0]:
#     print(f'File name: {result[0]}\nClip validity: {result[1][0]}\nClip length: {result[1][1]} seconds\n')