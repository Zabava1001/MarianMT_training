# MarianMT_training
This project is aimed at training the MarianMT model for translation from Khakass into Russian.

## Project Structure
    MarianMT_training/
    │
    ├── data/                # Папка для данных (например, dataset.xlsx)
    ├── model/               # Предустановленная модель для обучения
    ├── output-ru-ha/        # Папка для выходных данных (например, логов и результатов)
    ├── src/                 # Исходный код проекта
     │   ├── config.py        # Конфигурации и параметры
     │   ├── train.py          # Главный скрипт для тренировки модели
     │   ├── model.py         # Подготовка модели к обучению
     │   ├── dataset.py       # Загрузка и подготовка данных
     ├── bleu/            # Оценка модели на основе BLEU-оценки
     │   └── bleu_score.py
     ├── test_translation/ # Тестирование модели для перевода
     │       └── translation.py
     └── README.md            # Этот файл

## Model Information
The model **MarianMT** used for training will be stored in the "model/" directory, you can download it using the script:

    python downloading_model.py


### Installing PyTorch
Depending on your graphics card, install the appropriate version of PyTorch:

**For GPU (NVIDIA CUDA):**
    
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

### Install requirements
Before running the project, make sure to install all the required dependencies by running the following command:

    pip install -r requirements.txt


### Quick start
To start training the model, simply run the following command:

     python main.py

#### This command will:
1. Load the data (from data/dataset.xlsx).
2. Prepare the dataset for training.
3. Start the training process with the model and hyperparameters defined in src/config.py.

The training logs and model checkpoints will be saved in the output-ru-ha folder.

### Evaluation
Once the training is complete, you can evaluate the model by calculating the BLEU score. To do so, run the evaluation script (Once the training is complete, you can evaluate the model by calculating the BLEU score. This will calculate the BLEU score on a random sample from the test dataset and print the results. To do so, run the evaluation script) :

    python src/bleu/bleu_score.py

### Translation
You can also use the trained model to translate text. To do so, run the translation script (This script demonstrates how to use the trained model to translate Russian text into Khakass):

    python src/test_translation/translation.py

### Configuration
All hyperparameters and model settings are located in the src/config.py file. Some of the key parameters include:

<span style="background-color:gainsboro;">MAX_LENGTH</span>: Maximum sequence length for tokenization.

<span style="background-color:gainsboro;">SAMPLE_SIZE</span>: The size of the sample for BLEU score calculation.

<span style="background-color:gainsboro;">TEST_SIZE</span>: The size of the test split from the dataset.

Feel free to adjust these parameters based on your needs.
