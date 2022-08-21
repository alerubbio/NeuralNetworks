import os
VIDEO_FOLDER_PATH = 'E:\Projects\TwitchMontage\VideoCompilation\VideoFiles\\raw_clips'

def sort_folder(folder):
    # Get list of all files only in the given directory
    list_of_files = filter( lambda x: os.path.isfile(os.path.join(folder, x)),
                            os.listdir(folder))

    # Sort list of files based on last modification time in ascending order
    list_of_files = sorted( list_of_files,
                            key = lambda x: -os.path.getctime(os.path.join(folder, x)))
    return list_of_files
        
def rename_files(folder):

    video_folder = sort_folder(folder)
    for count, filename in enumerate(video_folder):
        dst = f"clip{str(count)}.mp4"
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}/{dst}"
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)


rename_files(VIDEO_FOLDER_PATH)