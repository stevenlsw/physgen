import os
from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(input_folder, output_folder=None):
    # If no output folder is specified, save GIFs in the input folder
    if output_folder is None:
        output_folder = input_folder

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            # Full path of the mp4 file
            mp4_path = os.path.join(input_folder, filename)
            
            # Base filename without extension
            base_filename = os.path.splitext(filename)[0]
            
            # Full path for the output gif file
            gif_path = os.path.join(output_folder, f"{base_filename}.gif")
            
            # Convert mp4 to gif with good quality settings
            try:
                clip = VideoFileClip(mp4_path)
                clip = clip.set_duration(clip.duration)  # Ensure full video is converted
                # clip = clip.resize(height=480)  # Resize the GIF, optional
                clip.write_gif(gif_path, fps=8, program='ffmpeg', loop=0)
                print(f"Successfully converted {mp4_path} to {gif_path}")
            except Exception as e:
                print(f"Error converting {mp4_path}: {e}")

# Replace 'path/to/your/folder' with the path to your folder containing .mp4 files
input_folder = 'assets'
convert_mp4_to_gif(input_folder)

