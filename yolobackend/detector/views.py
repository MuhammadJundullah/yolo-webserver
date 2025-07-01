# detector/views.py
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.http import FileResponse # Mungkin tidak perlu lagi import ini jika tidak langsung serve file
from .yolo_model import detect_objects_on_image, process_video_for_detection
from PIL import Image
import io
import os
import uuid
import shutil

os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

class ImageDetectionView(APIView):
    # ... (kode ini tetap sama) ...
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response(
                {"error": "No image provided in the request."},
                status=status.HTTP_400_BAD_REQUEST
            )

        image_file = request.FILES['image']

        try:
            image_bytes = image_file.read()
            detections = detect_objects_on_image(image_bytes)

            if "error" in detections:
                return Response(
                    detections,
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            return Response(detections, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": f"An unexpected error occurred during image detection: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class VideoDetectionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'video' not in request.FILES:
            return Response(
                {"error": "No video provided in the request."},
                status=status.HTTP_400_BAD_REQUEST
            )

        video_file = request.FILES['video']
        temp_dir = None

        try:
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_videos', str(uuid.uuid4()))
            os.makedirs(temp_dir, exist_ok=True)
            input_video_path = os.path.join(temp_dir, video_file.name)

            with open(input_video_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)

            output_video_path = process_video_for_detection(input_video_path, temp_dir)

            if output_video_path is None:
                # Bersihkan direktori sementara jika ada kesalahan
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                return Response(
                    {"error": "Failed to process video for detection: Could not open video writer."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # --- PERUBAHAN UTAMA DI SINI ---
            # Daripada mengirim FileResponse, kita kirim URL video hasil
            # Contoh: /media/temp_videos/uuid/detected_video.mp4
            # Perhatikan: URL ini bisa diakses karena kita sudah konfigurasi static(MEDIA_URL, document_root=MEDIA_ROOT)
            relative_output_path = os.path.relpath(output_video_path, settings.MEDIA_ROOT)
            download_url = f"{settings.MEDIA_URL}{relative_output_path}"

            # Bersihkan file input dan direktori sementara
            # Penting untuk membersihkan yang tidak lagi dibutuhkan agar tidak memakan ruang
            os.remove(input_video_path) # Hapus file input
            if os.path.exists(temp_dir) and not os.listdir(temp_dir): # Hapus direktori jika kosong
                os.rmdir(temp_dir)
            # Catatan: File output akan tetap ada di MEDIA_ROOT dan dapat diakses via URL

            return Response(
                {"message": "Video processed successfully!", "download_url": download_url},
                status=status.HTTP_200_OK
            )

        except Exception as e:
            print(f"Error during video detection: {e}")
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            return Response(
                {"error": f"An unexpected error occurred during video detection: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )