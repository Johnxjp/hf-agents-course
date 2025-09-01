"""
Required tools:
- Web Search: can use Google
- File Read
- Image Understanding
- Audio Understanding
"""

from pathlib import Path

from agents import function_tool
from dotenv import load_dotenv

from src.utils import read_text_file, read_image_file, read_audio_file, read_video_file

load_dotenv()


@function_tool
def read_file(file_path: str) -> str | bytes:
    """
    Reads the content of a file and returns it as a string or bytes.
    Type of file is inferred from suffix
    """
    path_uri = Path(file_path)
    if path_uri.suffix == ".txt" or path_uri.suffix == ".py":
        return read_text_file(file_path)
    elif path_uri.suffix in [".jpg", ".jpeg", ".png"]:
        return read_image_file(file_path)
    elif path_uri.suffix in [".mp3", ".wav"]:
        return read_audio_file(file_path)
    elif path_uri.suffix in [".mp4"]:
        return read_video_file(file_path)
    else:
        raise ValueError("Unsupported file type")
