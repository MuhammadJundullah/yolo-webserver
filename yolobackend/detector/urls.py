# detector/urls.py
from django.urls import path
from .views import ImageDetectionView, VideoDetectionView # Import View baru
from .consumers import GPUCheckView # Import GPUCheckView

urlpatterns = [
    path('detect/image/', ImageDetectionView.as_view(), name='image_detection'), # Ganti path ini
    path('detect/video/', VideoDetectionView.as_view(), name='video_detection'), # TAMBAHKAN BARIS INI
     path('gpu-check/', GPUCheckView.as_view(), name='gpu_check'),
]