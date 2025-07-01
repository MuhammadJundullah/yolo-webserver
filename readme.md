# How to Run

1. **Install Miniconda**  
    Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your operating system.

2. **Create the Conda Environment**  
    ```bash
    conda env create -f environment.yml
    ```

3. **Start the Server**  
    Run the following command in your terminal:
    ```bash
    DJANGO_SETTINGS_MODULE=yolobackend.settings daphne -b 0.0.0.0 -p 8000 yolobackend.asgi:application
    ```

4. **Open the Frontend**  
    Open `index.html` in your web browser.