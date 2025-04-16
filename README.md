# SeNER_Hindi: Small Language Model Driven Named Entity Recognition for Hindi
[Paper](https://arxiv.org/pdf/2502.07286) [Dataset](https://huggingface.co/datasets/ai4bharat/naamapadam)
## Overview
SeNER_Hindi is a Named Entity Recognition (NER) system designed for the Hindi language. It leverages a Small Language Model (SLM) to accurately identify entities such as persons, locations, and organizations from Hindi text. The project aims to enhance Natural Language Understanding (NLU) capabilities for Hindi, supporting downstream NLP applications like machine translation, sentiment analysis, and information retrieval.


## Methodology
SeNER_Hindi is based on the **SeNER** model, a lightweight span-based NER method with two key innovations:
1. **BiDirectional Arrow Attention + LogN-Scaling**: Combines local and global attention while stabilizing attention entropy for variable-length inputs.
2. **BiSPA (Bidirectional Sliding Window plus Attention)**: Reduces redundant span computations by compressing candidate spans and modeling interactions bidirectionally.

### System Architecture
1. **Input Encoding**: Uses a pre-trained language model (PLM) with arrow attention.
2. **Biaffine Model**: Generates span representations for candidate entities.
3. **BiSPA**: Compresses spans and models interactions horizontally and vertically.
4. **Training**: Trained on the Naamapadam dataset using binary cross-entropy loss.

## Results
- Trained on 10,000 samples from the Naamapadam dataset (Google Colab, T4 GPU).
- Achieved training loss: 0.2706, validation loss: 0.2488 (1 epoch).
- High accuracy in recognizing "Person" and "Location" entities.

## Future Work
1. Train on larger datasets and more epochs for better generalization.
2. Incorporate benchmark datasets like HiNER to reduce bias.
3. Use LoRA with PEFT for efficient fine-tuning.

## Installation and Usage
### Clone the Repository
1. Clone this repository to your local machine:
git clone https://github.com/your-username/SeNER_Hindi.git
cd SeNER_Hindi

3. Install the required dependencies using requirements.txt:
pip install -r requirements.txt

3. Execute the main script to start the NER system:
python main.py

## Acknowledgments
Special thanks to the authors of the SeNER paper and the Hugging Face community for their contributions to open-source NLP tools.
