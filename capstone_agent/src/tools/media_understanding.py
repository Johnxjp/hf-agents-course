from agents import function_tool

from google import genai
from google.genai import types as google_types
from typer import prompt

from src.utils import read_audio_file, read_image_file


client = genai.Client()

GOOGLE_MODEL = "gemini-2.0-flash"


@function_tool
def visual_question_answer(file_path: str, question: str) -> str:
    """
    Use to analyze an image and answer a question about it.
    Args:
        file_path (str): The path to the image file.
        question (str): The question to be answered.
    Returns:
        str: The answer to the question.
    """
    print("Calling function 'visual_question_answer'")
    print(f"Analyzing image file {file_path} to answer question: {question}")
    image_object = read_image_file(file_path)
    response = client.models.generate_content(
        model=GOOGLE_MODEL,
        contents=[
            image_object,
            question + " Respond briefly and directly to the question.",
        ],
    )
    result = response.text
    print(f"Response received {result}")
    return result


@function_tool
def generate_transcript(file_path: str) -> str:
    """
    Use to transcribe an audio or video file to text.
    Args:
        file_path (str): The path to the audio or video file.
    Returns:
        str: The transcript of the audio or video file.
    """
    print("Calling function 'generate_transcript'")
    print(f"Transcribing audio/video file {file_path}")
    media_object = read_audio_file(file_path)
    prompt = "Generate a transcript of the speech without timestamps."
    response = client.models.generate_content(
        model=GOOGLE_MODEL,
        contents=[
            prompt,
            google_types.Part.from_bytes(
                data=media_object,
                mime_type="audio/mp3",
            ),
        ],
    )
    transcript = response.text
    print(f"Response received {transcript}")
    return transcript


@function_tool
def transcribe_youtube_video(video_url: str) -> str:
    """
    Use to transcribe a YouTube video to text.
    Args:
        video_url (str): The URL of the YouTube video e.g. https://www.youtube.com/watch?v=1htKBjuUWec
    Returns:
        str: The transcript of the YouTube video.
    """
    print("Calling function 'transcribe_youtube_video'")
    print(f"Transcribing YouTube video {video_url}")
    prompt = f"Generate a transcript of the YouTube video at {video_url} without timestamps."
    response = client.models.generate_content(
        model=GOOGLE_MODEL,
        contents=[
            prompt,
            google_types.Part(file_data=google_types.FileData(file_uri=video_url)),
        ],
    )
    transcript = response.text
    print(f"Response received {transcript}")
    return transcript


@function_tool
def query_youtube_video(video_url: str, question: str) -> str:
    """
    Use to ask a question about the content in a YouTube video e.g. colour of a building.
    Args:
        video_url (str): The URL of the YouTube video e.g. https://www.youtube.com/watch?v=1htKBjuUWec
        question (str): The question to be answered.
    Returns:
        str: The answer to the question.
    """
    print("Calling function 'query_youtube_video'")
    print(f"Querying YouTube video {video_url} with question: {question}")
    prompt = question + " Respond briefly and directly to the question."
    response = client.models.generate_content(
        model=GOOGLE_MODEL,
        contents=[
            prompt,
            google_types.Part(file_data=google_types.FileData(file_uri=video_url)),
        ],
    )
    answer = response.text
    print(f"Response received {answer}")
    return answer
