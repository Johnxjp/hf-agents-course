from PIL import Image


def read_text_file(file_path: str) -> str:
    """
    Reads the content of a text file and returns it as a string.
    """
    with open(file_path) as file:
        return file.read()


def read_image_file(file_path: str) -> Image.Image:
    """
    Reads the content of an image file and returns it as an Image object.
    """
    return Image.open(file_path)


def read_audio_file(file_path: str) -> bytes:
    """
    Reads the content of an audio file and returns it as a bytes object.
    """
    with open(file_path, 'rb') as f:
        return f.read()


def read_video_file(file_path: str) -> str:
    """
    Reads the content of a video file and returns it as a string.
    """
    pass
