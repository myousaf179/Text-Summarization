# Psychological Text Summarization

This repository contains a Jupyter Notebook that implements a text summarization pipeline using the Psychological Text Summarization Dataset. The primary objective is to fine-tune a T5 encoder with a custom TensorFlow decoder (incorporating LSTM and Multi-Head Attention) to generate concise summaries of psychological texts.

##  Features

- **Data Loading & Exploration**  
  Load and inspect the dataset from an Excel file, remove duplicates, and handle missing values.
- **Text Preprocessing**  
  Clean raw text by stripping HTML tags, punctuation, and stopwords for improved model performance.
- **Tokenization**  
  Use `T5Tokenizer` to convert text and summary inputs into token IDs, with inputs padded/truncated to fixed lengths.
- **Model Architecture**  
  - **Encoder**: Pre-trained `TFT5EncoderModel` from Hugging Face ("t5-base").  
  - **Decoder**: Custom TensorFlow decoder combining LSTM cells and Multi-Head Attention layers.  
- **Configuration**  
  Centralized hyperparameters in a `Config` class (batch sizes, epochs, learning rate, seed) for easy experimentation.
- **Train-Test Split**  
  Split normalized features into training and testing sets using `random_state=4` for reproducibility.
- **Training & Visualization**  
  Train the model, monitor loss curves, and visualize training vs. validation loss.
- **Evaluation**  
  Compute ROUGE-1, ROUGE-2, and ROUGE-L scores to assess summary quality.
- **Model Persistence**  
  Save both the fine-tuned model and tokenizer to Google Drive for reuse.

##  Notebook Structure

1. **Imports & Setup**: Install required packages and import libraries (NumPy, pandas, TensorFlow, Transformers, NLTK, etc.).  
2. **Configuration Class**: Define hyperparameters and random seed in one place.  
3. **Dataset Loading**: Read the Excel dataset and perform initial cleaning.  
4. **Text Cleaning**: Remove HTML tags, non-alphanumeric characters, and stopwords.  
5. **Tokenization**: Prepare inputs (`input_ids`, `attention_mask`) and decoder inputs.  
6. **Model Definition**: Build the encoder–decoder architecture.  
7. **Training**: Train for a set number of epochs, track loss, and plot curves.  
8. **Evaluation**: Implement ROUGE scoring and report average precision, recall, and F-measure.  
9. **Save Artifacts**: Mount Google Drive, specify a save directory, and store the model/tokenizer.

##  Installation & Usage

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/psych-text-summarization.git
   cd psych-text-summarization
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook**  
   - Open `text_summarizer.ipynb` in Jupyter or Google Colab.  
   - Execute cells in order.  
   - When prompted, mount Google Drive to save the trained model.

##  Results

- **Training Curves**: Visualization of training vs. validation loss over epochs.  
- **ROUGE Scores**: Average ROUGE-1, ROUGE-2, and ROUGE-L metrics on the test set.

> _Example:_ `ROUGE-1 F-measure: 0.45`, `ROUGE-2 F-measure: 0.22`, `ROUGE-L F-measure: 0.40`  
_(Actual values may vary based on hyperparameters.)_



