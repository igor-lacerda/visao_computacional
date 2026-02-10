import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def resize_single_image(
    image: np.ndarray, 
    scale_factor: float
) -> np.ndarray:
    """
    Aplica o redimensionamento em uma imagem (array numpy).

    :param image: Array numpy da imagem de entrada.
    :param scale_factor: Fator de escala do redimensionamento (ex: 1.25 para 25% maior etc.).
    """

    if image is None:
        raise ValueError("Imagem inválida ou None.")

    height, width = image.shape[:2]

    # Casting para int porque as dimensões de pixel não admitem valores decimais
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    return resized_image

def detect_edges_single_image(
    image: np.ndarray,
    low_threshold: int = 50, 
    high_threshold: int = 100, 
    blur_kernel_size: int = 5,
    inverted: bool = True
) -> np.ndarray:
    """
    Retorna as bordas de uma imagem fornecida.

    :param image: Array numpy da imagem de entrada.
    :param low_threshold: Limite inferior para a histerese do Canny.
    :param high_threshold: Limite superior para a histerese do Canny.
    :param blur_kernel_size: Tamanho do kernel para o GaussianBlur (deve ser ímpar).
    :param inverted: Inverte as cores das bordas e do fundo da imagem de saída.
    """

    if image is None:
        raise ValueError("Imagem inválida ou None.")

    # Verifica se a imagem fornecida é cinza ou não
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Filtro passa-baixas (reduz ruídos)
    blurred = cv2.GaussianBlur(image_gray, (blur_kernel_size, blur_kernel_size), 0)

    edges = cv2.Canny(blurred, threshold1=low_threshold, threshold2=high_threshold)
    
    if inverted: 
        edges = cv2.bitwise_not(edges)
    
    return edges

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
        # print(f"Vídeo salvo em: {output_folder}")

def extract_frames_from_video(
    video: str, 
    output_dir: str = None
) -> None:
    """
    Salva todos os frames de um vídeo em png em um diretório fornecido.

    :param video_path: Caminho do vídeo.
    :param output_dir: Diretório em que serão salvas as imagens. Se None, cria uma pasta no diretório do vídeo de entrada.
    """
    video_path = Path(video)

    # Configura o diretório de saída das imagens
    output_folder = Path(output_dir) if output_dir else video_path.parent / "video_frames"
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Caminho '{video_path}' inválido.")
        return

    # Contador de frames
    frame_index = 1
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret: break

        cv2.imwrite(str(output_folder / f"frame_{frame_index:04d}.png"), frame)
        frame_index += 1

    cap.release()

def process_video_edges(
    video, 
    low_threshold: int = 50, 
    high_threshold: int = 100, 
    blur_kernel_size: int = 5,
    inverted: bool = True,
    output_dir: str = None
) -> None:
    """
    Reconhece as bordas de cada frame de um vídeo e retorna um outro de saída feito com essas imagens.

    :param video: Caminho do vídeo de entrada.
    :param low_threshold: Limite inferior para a histerese do Canny.
    :param high_threshold: Limite superior para a histerese do Canny.
    :param blur_kernel_size: Tamanho do kernel para o GaussianBlur (deve ser ímpar).
    :param inverted: Inverte as cores das bordas e do fundo da imagem de saída.
    :param output_dir: Pasta de destino do vídeo de saída. Se None, cria uma pasta no diretório do vídeo fornecido.
    """
    video_path = Path(video)

    # Configura o diretório de saída das imagens
    output_folder = Path(output_dir) if output_dir else video_path.parent / "video_borders"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_filename = f"{video_path.stem}_edges.mp4"
    output_filepath = output_folder / output_filename

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

            processed_frame = detect_edges_single_image(frame, low_threshold, high_threshold, blur_kernel_size, inverted)
            frame_out = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

            video_writer.write(frame_out)

    except KeyboardInterrupt:
        print("\nProcessamento interrompido pelo usuário.")
    except Exception as e:
        print(f"\nErro durante o processamento: {e}")
    finally:
        cap.release()
        video_writer.release()
        # print(f"Vídeo salvo em: {output_folder}")

def concatenate_videos(
    video_1: str, 
    video_2: str,
    orientation: str = "horizontal",  
    output_dir: str = None
) -> None:
    """
    Cria um vídeo a partir de outros dois, juntando-os lado a lado horizontal ou verticalmente.

    :param video_1: Caminho do primeiro vídeo de entrada.
    :param video_2: Caminho do segundo.
    :param orientation: Determina se o vídeo terá orientação vertical ou horizontal. Se não informado, o padrão é horizontal.
    :param output_dir: Pasta de destino do vídeo de saída. Se None, cria uma pasta no diretório do vídeo fornecido.
    """
    if orientation.lower() not in ("horizontal", "vertical", "h", "v"):
        raise ValueError("Valor de 'orientation' deve ser 'Horizontal' ou 'Vertical'")

    video_path_1 = Path(video_1)
    video_path_2 = Path(video_2)

    # Configura o diretório de saída das imagens
    output_folder = Path(output_dir) if output_dir else video_path_1.parent / "merged_videos"
    output_folder.mkdir(parents=True, exist_ok=True)

    suffix = "h" if orientation in ("horizontal", "h") else "v"
    output_filename = f"{video_path_1.stem}_{video_path_2.stem}_merged_{suffix}.mp4"
    output_filepath = output_folder / output_filename

    # Inicializa as capturas
    cap_1 = cv2.VideoCapture(str(video_path_1))
    cap_2 = cv2.VideoCapture(str(video_path_2))

    if not cap_1.isOpened() or not cap_2.isOpened():
        print(f"Caminho '{video_path_1}' ou '{video_path_2}' inválido.")
        return

    # Considera o vídeo com menor taxa de quadros
    fps = min(cap_1.get(cv2.CAP_PROP_FPS), cap_2.get(cv2.CAP_PROP_FPS))

    # Extração de propriedades dos vídeos
    width_1 = int(cap_1.get(cv2.CAP_PROP_FRAME_WIDTH))
    width_2 = int(cap_2.get(cv2.CAP_PROP_FRAME_WIDTH)) 

    height_1 = int(cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height_2 = int(cap_2.get(cv2.CAP_PROP_FRAME_HEIGHT))  

    # Tratamento dos tamanhos dos vídeos
    if orientation.lower() in ("horizontal", "h"):
        # Horizontalmente, a "altura" de ambos os vídeos deve ser a mesma
        if height_1 != height_2:
            scale = height_1 / height_2
            new_width_2 = int(width_2 * scale)
            new_height_2 = height_1

            size_2 = (new_width_2, new_height_2)
        else:
            size_2 = (width_2, height_2)
        
        final_size = (width_1 + size_2[0], size_2[1])
    
    else:
        # Verticalmente, a "largura" deve ser igual
        if width_1 != width_2:
            scale = width_1 / width_2
            new_width_2 = width_1
            new_height_2 = int(height_2 * scale)

            size_2 = (new_width_2, new_height_2)
        else:
            size_2 = (width_2, height_2)
        
        final_size = (size_2[0], height_1 + size_2[1])

    video_writer = cv2.VideoWriter(str(output_filepath), cv2.VideoWriter_fourcc(*"mp4v"), fps, final_size)

    try:
        while True:
            ret1, frame_1 = cap_1.read()
            ret2, frame_2 = cap_2.read()

            if not ret1 or not ret2: break

            # Redimensiona o frame 2 se necessário
            if size_2 != (width_2, height_2):
                frame_2 = cv2.resize(frame_2, size_2)

            if orientation.lower() in ("horizontal", "h"):
                merged_frame = cv2.hconcat([frame_1, frame_2])
            else:
                merged_frame = cv2.vconcat([frame_1, frame_2])

            video_writer.write(merged_frame)
            
    except KeyboardInterrupt:
        print("\nProcessamento interrompido pelo usuário.")
    except Exception as e:
        print(f"\nErro durante o processamento: {e}")
    finally:
        cap_1.release()
        cap_2.release()
        video_writer.release()

def resize_images_from_directory(
    source_dir: str, 
    scale_factor: float, 
    output_dir: str = None
) -> None:
    """
    Redimensiona todas as imagens de um diretório.
    
    :param source_dir: Caminho da pasta contendo as imagens.
    :param scale_factor: Fator de escala do redimensionamento (ex: 1.25 para 25% maior etc.).
    :param output_dir: É a pasta de destino. Se None, cria uma pasta "resized" dentro da origem.
    """

    input_path = Path(source_dir)

    # Configura o diretório de saída das imagens
    save_path = Path(output_dir) if output_dir else input_path / "resized_results"
    save_path.mkdir(parents=True, exist_ok=True)

    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")

    for file in input_path.iterdir():
        if file.suffix.lower() in valid_extensions:
            image = cv2.imread(str(file))

            if image is None:
                print(f"Arquivo corrompido ou inválido: {file.name}")
                continue
            
            try:
                processed_image = resize_single_image(image, scale_factor)

                output_file = save_path / file.name
                cv2.imwrite(str(output_file), processed_image)
            except Exception as e:
                print(f"Erro ao processar {file.name}: {e}")

def detect_edges_from_directory(
    source_dir: str, 
    low_threshold: int = 50, 
    high_threshold: int = 100, 
    blur_kernel_size: int = 5,
    inverted: bool = True,
    output_dir: str = None
) -> None:
    """
    Detecta bordas em imagens, como um sketch, de um diretório usando o algoritmo Canny.
    
    :param source_dir: Caminho da pasta de origem.
    :param low_threshold: Limite inferior para a histerese do Canny.
    :param high_threshold: Limite superior para a histerese do Canny.
    :param blur_kernel_size: Tamanho do kernel para o GaussianBlur (deve ser ímpar).
    :param output_dir: Pasta de destino. Se None, cria "edge_results" no mesmo diretório.
    """
    input_path = Path(source_dir)

    # Configura o diretório de saída das imagens
    save_path = Path(output_dir) if output_dir else input_path / "edge_results"
    save_path.mkdir(parents=True, exist_ok=True)

    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")

    for file in input_path.iterdir():
        if file.suffix.lower() in valid_extensions:
            image = cv2.imread(str(file))

            if image is None:
                print(f"Arquivo corrompido ou inválido: {file.name}")
                continue
            
            try: 
                processed_image = detect_edges_single_image(image, low_threshold, high_threshold, blur_kernel_size, inverted)
                
                output_file = save_path / file.name
                cv2.imwrite(str(output_file), processed_image)
            except Exception as e:
                print(f"Erro ao processar {file.name}: {e}")

if __name__ == "__main__":
    print("Este arquivo não deve ser executado diretamente, ao invés disso execute o arquivo 'main.py'")

