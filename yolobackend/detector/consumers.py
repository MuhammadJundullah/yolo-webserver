# detector/consumers.py
import json
import base64
import numpy as np
import cv2
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from ultralytics import YOLO
import torch # Untuk deteksi GPU

# Muat model YOLO sekali saat server dimulai
try:
    model = YOLO('yolov8n.pt')
    print("YOLOv8nano model loaded for WebSocket consumer.")
    # Pindahkan model ke GPU jika tersedia
    if torch.cuda.is_available():
        model.to('cuda')
        print("YOLOv8nano model moved to CUDA (GPU).")
    else:
        print("YOLOv8nano model running on CPU (no CUDA GPU detected).")
except Exception as e:
    print(f"Error loading YOLOv8nano model for WebSocket: {e}")
    model = None

class VideoDetectionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        if model is None:
            print("WebSocket connection denied: YOLO model not loaded.")
            await self.close()
            return

        # Anda bisa menambahkan logika otentikasi di sini jika diperlukan
        # if not self.scope["user"].is_authenticated:
        #     await self.close()
        #     return

        await self.accept()
        print(f"WebSocket connected from {self.scope['client']}")

    async def disconnect(self, close_code):
        print(f"WebSocket disconnected from {self.scope['client']} with code {close_code}")

    async def receive(self, text_data=None, bytes_data=None):
        # Menerima frame dari frontend
        if bytes_data:
            # Konversi bytes gambar ke numpy array (OpenCV format)
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Failed to decode frame from bytes.")
                return

            # Jalankan inferensi YOLO
            # Ultralytics model() bisa menerima numpy array langsung
            # Pastikan model di GPU jika tersedia
            try:
                if torch.cuda.is_available():
                    results = model(frame, conf=0.25, iou=0.7, device='cuda') # Pastikan inferensi di GPU
                else:
                    results = model(frame, conf=0.25, iou=0.7, device='cpu') # Inferensi di CPU

                detections_list = []
                for r in results:
                    # Konversi tensor PyTorch ke numpy array di CPU
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy()

                    for box, conf, cls in zip(boxes, confs, clss):
                        x1, y1, x2, y2 = map(int, box)
                        class_name = model.names[int(cls)]
                        detections_list.append({
                            "box": [x1, y1, x2, y2],
                            "confidence": float(conf),
                            "class": class_name
                        })

                # Kirim hasil deteksi kembali ke frontend
                await self.send(text_data=json.dumps({
                    "detections": detections_list,
                    "frame_shape": [frame.shape[1], frame.shape[0]] # [width, height]
                }))
            except Exception as e:
                print(f"Error during real-time detection: {e}")
                await self.send(text_data=json.dumps({"error": f"Detection error: {e}"}))

# --- API endpoint untuk cek GPU (HTTP Request biasa) ---
from rest_framework.views import APIView
from rest_framework.response import Response
import torch # Pastikan torch sudah diimport

class GPUCheckView(APIView):
    def get(self, request, *args, **kwargs):
        gpu_available = torch.cuda.is_available()
        return Response({"gpu_available": gpu_available})