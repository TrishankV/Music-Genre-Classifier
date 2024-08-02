# Music-Genre-Classifier

This repository contains a music genre classification model that achieves an accuracy of 86%. The model is trained on the GZNAT dataset and uses a simple Streamlit app for user interaction.

## Model and Dataset

- **Model Accuracy**: 86%
- **Training Platform**: Google Colab
- **Dataset**: GZNAT Dataset [Link](https://github.com/user-attachments/assets/da65b241-502e-42c5-96cc-a4d593f7d93a)

## Quick Links

- [Hugging Face Model](https://huggingface.co/TrishankV/Music-Genre-Classifier-78.5acc/tree/main)
- [Google Colab Notebook](https://colab.research.google.com/drive/1jyAP5FgSfu0VRrZZ6Rv7lofRYghRFmRF?usp=share_link)

## Running the Streamlit App

To run the Streamlit app for music genre classification, follow these steps:

1. **Clone the Repository**
    ```bash
    git clone https://github.com/YourUsername/Music-Genre-Classifier.git
    cd Music-Genre-Classifier
    ```

2. **Install the Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```

### Example Outputs

1. **Turkish March - Classical**
    ![Turkish March](https://github.com/user-attachments/assets/356a3100-8ad0-4f7c-b106-bffcef728e65)

2. **Confusion Between Rock and Metal**
    ![Rock and Metal](https://github.com/user-attachments/assets/2df7f439-c59e-445c-9078-a0d55b199a0a)

3. **Incorrect Classification of "Believer" as Blues**
    ![Believer](https://github.com/user-attachments/assets/070e03f4-fb37-4aec-8438-ba1c9c96b86a)

## Screenshots

Below are some screenshots of the Streamlit app in action:

![Screenshot 1](https://github.com/user-attachments/assets/e6800fb7-e16e-472b-b6a4-f6819efbe43a)
![Screenshot 2](https://github.com/user-attachments/assets/fb0c0b69-ab4a-45c5-94cf-2beaf8aa4b80)

## Known Issues

- The model sometimes confuses similar genres, such as rock and metal.
- Certain songs, like "Believer," are incorrectly classified (e.g., as blues).
