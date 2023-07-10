# Music Visualization Network: Towards Audio to Image Synthesis using cDCGAN

This project aims to develop a Music Visualization Network (MVNet) that generates visual representations of music from various art movements in History of Art. The main goal is to establish a coherent and meaningful mapping between the auditory and visual modalities, enabling the exploration of intricate semantic relationships that might potentially exist between music and paintings throughout different time periods. By employing advanced techniques in deep learning and cross-modal analysis, the MVNet enhances our understanding of how music and paintings relate to each other, offering new ways to perceive and appreciate their semantic correlation.

**In this part of the project we:**
* implemented the Convolutional Audio Encoder, which reduces the dimensionality of the audio data and extracts meaningful features for mapping to image representations.
* developed the Conditional Deep-Convolutional GAN (cDCGAN) model, which generates visually coherent images that align with the style of the input audio.
* trained the MVNet using the training dataset consisting of paired audio and image samples.
* fine-tuned the MVNet by optimizing the generator and discriminator models using the Adam optimizer with specific configurations.
* conducted experiments to evaluate the performance of the MVNet on the testing dataset, which contains audio and image pairs for different art periods.
* analyzed the generated images to assess the similarity between the generated and pair images, providing insights into the effectiveness of the MVNet in capturing the style and characteristics of the input audio.

## Prerequisites
The following python packages are required for the code to run:
* Python 3: https://www.python.org/downloads/
* tqdm: `pip install tqdm`
* Torch: `pip install torch`
* Matplotlib: `pip install matplotlib`
* NumPy: `pip install numpy`
* Librosa: `pip install librosa`
* Torchvision: `pip install torchvision`

**Alternatively:** you can download [requirements.txt](https://github.com/nataliakoliou/Music-Visualization-Network/blob/main/requirements.txt) and run ```pip install -r requirements.txt```, to automatically install all the packages needed to reproduce my project on your own machine.

  ## How to Run the Code
  1) Access the Musart-Dataset in Google Drive through this [link](https://github.com/nataliakoliou/Music-Visualization-Network/musart-dataset.txt) and create an instance of it in your own Google Drive under a folder named "DATASETS".
  2) Create a folder named "MODELS" in your Google Drive and place these [pre-trained models](https://github.com/nataliakoliou/Music-Visualization-Network/pretrained-models.txt).
  3) Place the following files somewhere in your Drive: [_utils.py](https://github.com/nataliakoliou/Music-Visualization-Network/_utils.py), [_models.py](https://github.com/nataliakoliou/Music-Visualization-Network/_models.py), [_train.py](https://github.com/nataliakoliou/Music-Visualization-Network/_train.py), [_test.py](https://github.com/nataliakoliou/Music-Visualization-Network/_test.py), and [main.py](https://github.com/nataliakoliou/Music-Visualization-Network/main.py).
  4) To run the code, execute the main.py script. You can use Google Colab for this purpose. Follow the instructions provided in the console. If you want to train the MVNet, type "train" when prompted. If you want to test the MVNet, type "test".
  5) During the process (either training or testing), you will be asked to confirm whether you already have the training.npy and testing.npy data files in your Drive under a folder named "DATA". If you have these files, type "y" (yes). If you don't have them, type "n" (no), and the code will create them for you. Make sure to create the "DATA" folder in your Drive if it doesn't exist.

## Author
Natalia Koliou: find me on [LinkedIn](https://www.linkedin.com/in/natalia-k-b37b01197/).
