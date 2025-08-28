import ffmpeg as ffmpeg_binaries

# Initialize the module (downloads binaries if not found)
ffmpeg_binaries.init()

# Get the path to the ffmpeg executable
FFMPEG_PATH = ffmpeg_binaries.FFMPEG_PATH