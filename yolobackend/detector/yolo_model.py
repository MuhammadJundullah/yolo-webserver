# detector/yolo_model.py
import torch
from ultralytics import YOLO
import cv2 # Import OpenCV
import numpy as np # OpenCV sering mengembalikan array numpy
import os
import uuid # Untuk membuat nama file unik

# Memuat model YOLOv8nano.
try:
    model = YOLO('best.pt')
    print("YOLOv8nano model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8nano model: {e}")
    model = None

# Fungsi deteksi objek untuk gambar (dari sebelumnya, tetap ada)
def detect_objects_on_image(image_input):
    # ... (kode fungsi ini sama seperti sebelumnya, tidak perlu diubah) ...
    from PIL import Image # Import Pillow di sini jika hanya digunakan di fungsi ini
    import io

    if model is None:
        return {"error": "YOLOv8nano model not loaded. Check server logs."}

    if isinstance(image_input, bytes):
        try:
            image = Image.open(io.BytesIO(image_input))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            return {"error": f"Failed to open/decode image bytes using PIL: {e}"}
    elif isinstance(image_input, Image.Image):
        image = image_input
        if image.mode != 'RGB':
            image = image.convert('RGB')
    else:
        return {"error": "Unsupported input type for detection. Expected bytes or PIL.Image."}

    try:
        results = model(image, conf=0.25, iou=0.7)

        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls)]
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class": class_name
                })
        return detections
    except Exception as e:
        print(f"Error during object detection: {e}")
        return {"error": f"Failed to process image for detection with YOLO: {e}"}


# --- FUNGSI BARU UNTUK DETEKSI VIDEO ---
def process_video_for_detection(input_video_path, output_dir, conf_threshold=0.25, iou_threshold=0.7):
    if model is None:
        print("Error: YOLOv8nano model not loaded.")
        return None

    cap = cv2.VideoCapture(input_video_path, cv2.CAP_AVFOUNDATION) # Coba AVFoundation dulu
    if not cap.isOpened():
        print(f"Warning: Could not open video with AVFoundation, trying default backend for {input_video_path}")
        cap = cv2.VideoCapture(input_video_path) # Coba default (GStreamer atau lainnya)

    if not cap.isOpened():
        print(f"Error: Still could not open video file {input_video_path} with any backend.")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # --- STRATEGI COBA-COBA FORMAT/CODEC ---
    output_video_path = None
    tried_formats = [
        {'fourcc': cv2.VideoWriter_fourcc(*'mp4v'), 'ext': '.mp4', 'codec_name': 'MP4V (MP4)'}, # Prioritas 1: MP4
        {'fourcc': cv2.VideoWriter_fourcc(*'MJPG'), 'ext': '.avi', 'codec_name': 'MJPG (AVI)'}, # Prioritas 2: AVI/MJPG (sering lebih kompatibel)
        {'fourcc': cv2.VideoWriter_fourcc(*'XVID'), 'ext': '.avi', 'codec_name': 'XVID (AVI)'}, # Prioritas 3: AVI/XVID
    ]

    out = None
    for fmt in tried_formats:
        current_output_filename = f"detected_video_{uuid.uuid4().hex}{fmt['ext']}"
        current_output_path = os.path.join(output_dir, current_output_filename)

        print(f"Attempting to open video writer with {fmt['codec_name']} to {current_output_path}")
        out = cv2.VideoWriter(current_output_path, fmt['fourcc'], fps, (frame_width, frame_height))

        if out.isOpened():
            output_video_path = current_output_path
            print(f"Successfully opened video writer with {fmt['codec_name']}.")
            break # Berhenti jika berhasil
        else:
            print(f"Failed to open video writer with {fmt['codec_name']}.")
            if os.path.exists(current_output_path):
                os.remove(current_output_path) # Hapus file kosong jika gagal

    if out is None or not out.isOpened():
        print("Error: Could not open video writer with any supported codec/format.")
        cap.release()
        return None
    # --- AKHIR STRATEGI COBA-COBA ---

    frame_count = 0
    print(f"Starting video processing: {input_video_path}")
    print(f"  Frame resolution: {frame_width}x{frame_height}, FPS: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = model(frame, conf=conf_threshold, iou=iou_threshold)

            for r in results:
                for box_data in r.boxes:
                    x1, y1, x2, y2 = map(int, box_data.xyxy[0].cpu().numpy())
                    conf = box_data.conf[0].cpu().numpy()
                    cls = int(box_data.cls[0].cpu().numpy())
                    class_name = model.names[cls]

                    label = f"{class_name} {conf:.2f}"
                    color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label)*10 + 5, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            out.write(frame)

        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            out.write(frame) # Tetap tulis frame asli jika ada error deteksi

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Finished video processing. Output saved to: {output_video_path}")
    return output_video_path