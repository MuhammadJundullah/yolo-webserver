# how to run

1. install miniconda 
2. conda install requirements
3. run on terminal "DJANGO_SETTINGS_MODULE=yolobackend.settings daphne -b 0.0.0.0 -p 8000 yolobackend.asgi:application"