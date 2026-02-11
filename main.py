from utils.object_detection import detect_objects_yolo_video
from utils.image_processing import concatenate_videos
from pathlib import Path

def main():
    # Exemplo
    folder = r"C:\Users\Administrator\Desktop"
    video = "exemplo.mp4"

    # Parte 1
    folder_path = Path(folder)
    video_path = folder_path / "exemplo" / video

    detect_objects_yolo_video(video_path, detect_person=True, detect_vehicles=True)

    # Parte 2
    video_path_2 = folder_path / "exemplo" / "yolo_recognition" / "exemplo_detected.mp4"

    concatenate_videos(video_path, video_path_2, orientation='h')
    concatenate_videos(video_path, video_path_2, orientation='v')

if __name__ == "__main__":
    main()
