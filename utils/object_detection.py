import cv2
import numpy as np
from pathlib import Path

def _get_class_groups(
    person, 
    vehicle, 
    animal, 
    electronic
):
    """
    Gera um dicionário de configurações das classes com base em variáveis booleanas.
    É usado para organizar o código das funções a seguir que implementam YOLO.
    """
    return {
        "Person": {
            "ids": {0}, 
            "color": (0, 0, 255), # Vermelho
            "active": person
        },
        "Vehicle": {
            "ids": {1, 2, 3, 4, 5, 6, 7, 8}, 
            "color": (255, 0, 0), # Azul
            "active": vehicle
        },
        "Animal": {
            "ids": {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}, 
            "color": (0, 255, 255), # Amarelo
            "active": animal
        },
        "Electronic": {
            "ids": {63, 64, 65, 66, 67},
            "color": (255, 0, 255), # Roxo
            "active": electronic
        }
    }

def _process_frame(
    frame: np.ndarray, 
    model, 
    class_groups: dict,
    conf: float
) -> np.ndarray:
    """
    Recebe uma imagem, aplica o YOLO e desenha as caixas de detecção.
    """
    results = model(frame, stream=True, verbose=False, conf=conf)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])

            target_group = None
            for label, config in class_groups.items():
                if config["active"] and cls_id in config["ids"]:
                    target_group = (label, config["color"])
                    break
            
            if target_group is None:
                continue
            
            # Desenhos na imagem
            label_text, color = target_group
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
    return frame

def detect_objects_yolo_image(
    image: np.ndarray, 
    model_name: str = "yolov10n.pt",
    detect_person: bool = True,
    detect_vehicles: bool = False,
    detect_animals: bool = False,
    detect_electronics: bool = False,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """
    Processa e salva um vídeo aplicando detecção de objetos com YOLOv8.

    :param image: Imagem de entrada.
    :param model_name: Nome ou caminho do modelo (por exemplo, "yolov8n.pt").
    :param detect_person: Se True, detecta pessoas.
    :param detect_vehicles: Se True, detecta veículos (Carros, motos, ônibus, etc.).
    :param detect_animals: Se True, detecta animais.
    :param detect_electronics: Se True, detecta celulares, laptops, entre outros.
    :param confidence_threshold: Limiar de confiança (0.0 a 1.0) para considerar uma detecção.
    """
    from ultralytics import YOLO
    
    if image is None:
        raise ValueError("Imagem inválida ou None.")
    
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Erro crítico ao carregar modelo YOLO: {e}")
        return
    
    groups = _get_class_groups(detect_person, detect_vehicles, detect_animals, detect_electronics)

    processed_image = _process_frame(image, model, groups, confidence_threshold)

    return processed_image

def detect_objects_yolo_video(
    video, 
    model_name: str = "yolov10n.pt",
    detect_person: bool = True,
    detect_vehicles: bool = False,
    detect_animals: bool = False,
    detect_electronics: bool = False,
    confidence_threshold: float = 0.5,
    output_dir: str = None
) -> None:
    """
    Processa e salva um vídeo aplicando detecção de objetos com YOLOv8.

    :param video: Caminho do vídeo de entrada.
    :param model_name: Nome ou caminho do modelo (por exemplo, "yolov8n.pt").
    :param detect_person: Se True, detecta pessoas.
    :param detect_vehicles: Se True, detecta veículos (Carros, motos, ônibus, etc.).
    :param detect_animals: Se True, detecta animais.
    :param detect_electronics: Se True, detecta celulares, laptops, entre outros.
    :param confidence_threshold: Limiar de confiança (0.0 a 1.0) para considerar uma detecção.
    :param output_dir: Pasta de destino. Se None, cria uma pasta no diretório do vídeo fornecido.
    """
    from ultralytics import YOLO
    
    if confidence_threshold < 0 or confidence_threshold > 1:
        raise ValueError("Limiar de confiança fora do escopo entre 0 e 1")

    video_path = Path(video)

    # Configura o diretório de saída das imagens
    output_folder = Path(output_dir) if output_dir else video_path.parent / "yolo_recognition"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_filename = f"{video_path.stem}_detected.mp4"
    output_filepath = output_folder / output_filename

    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Erro crítico ao carregar modelo YOLO: {e}")
        return

    groups = _get_class_groups(detect_person, detect_vehicles, detect_animals, detect_electronics)

    # Processamento do vídeo
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Caminho '{video_path}' inválido.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    video_writer = cv2.VideoWriter(str(output_filepath), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret: break

            processed_frame = _process_frame(frame, model, groups, confidence_threshold)
            
            video_writer.write(processed_frame)

    except KeyboardInterrupt:
        print("\nProcessamento interrompido pelo usuário.")
    except Exception as e:
        print(f"\nErro durante o processamento: {e}")
    finally:
        cap.release()
        video_writer.release()

if __name__ == "__main__":
    print("Este arquivo não deve ser executado diretamente, ao invés disso execute o arquivo 'main.py'")
