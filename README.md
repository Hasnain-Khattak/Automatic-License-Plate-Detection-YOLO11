# License Plate Detection and Text Extraction

This project is a **License Plate Detection and Text Extraction** application built using YOLOv8 and Streamlit. The application detects license plates in images and videos, draws bounding boxes around them, and extracts the text from the detected license plates using Tesseract OCR.

## Features
- Detect license plates in images and videos.
- Draw bounding boxes around detected license plates.
- Extract text from detected license plates using Tesseract OCR.
- User-friendly interface created with Streamlit.

## Demo

![App Screenshot](https://github.com/Hasnain-Khattak/Automatic-License-Plate-Detection-YOLO11/blob/main/Screenshot%202025-01-25%20175053.png)

----

[Click here to watch the Demo Video](https://raw.githubusercontent.com/Hasnain-Khattak/Automatic-License-Plate-Detection-YOLO11/main/output/demo.mp4)



## Installation

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR installed ([Download Tesseract](https://github.com/tesseract-ocr/tesseract))

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/license-plate-detection.git
   cd license-plate-detection
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Tesseract OCR:
   - Install Tesseract OCR ([Download Tesseract](https://github.com/tesseract-ocr/tesseract))
   - Add the Tesseract executable path to your code:
     ```python
     pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
     ```

5. Download the YOLO model weights:
   - Place your YOLO model weights (e.g., `best.pt`) in the project directory.
   - Update the path in the code:
     ```python
     MODEL_PATH = "path_to_your_model/best.pt"
     ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Upload an image or video file through the interface.
3. The application will:
   - Detect license plates.
   - Draw bounding boxes around the detected plates.
   - Extract and display the text from the plates.
4. Processed media and extracted text will be displayed in the application.

## Directory Structure
```
license-plate-detection/
|
|-- app.py                  # Main application file
|-- requirements.txt        # Python dependencies
|-- best.pt                 # YOLO model weights (add this file manually)
|-- temp/                   # Temporary directory for uploaded files
|-- output/                 # Directory for processed media
|-- README.md               # Project documentation
```

## Requirements
- **Python Libraries**:
  - `streamlit`
  - `opencv-python`
  - `numpy`
  - `Pillow`
  - `ultralytics`
  - `pytesseract`

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Video Not Displaying in Streamlit
- Ensure the output video is encoded in MP4 with H.264 codec.
- Add a delay after writing the video to ensure it is saved properly.
- Verify the video file is playable in a local video player.

### Tesseract OCR Issues
- Ensure Tesseract is installed and the path is correctly set in the code.
- Test Tesseract OCR functionality separately to confirm it's working.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any feature additions or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [YOLOv8](https://github.com/ultralytics/yolov8) for object detection.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text extraction.
- [Streamlit](https://streamlit.io/) for building an interactive user interface.

## Contact
For questions or suggestions, feel free to open an issue or reach out via email at `msaqibkhan987987@gmail.com`.

---

Thank you for checking out this project! 🚗

