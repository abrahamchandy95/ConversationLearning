# ConversationScorer

ConversationScorer is a machine learning project for scoring conversations between a bot and a user. 
It uses data from Supabase along with target scores provided in a CSV file to train a regression model that predicts various conversational performance metrics.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)


## Introduction

In modern personalized learning systems, measuring the quality of conversations between an AI bot and a student can help improve engagement and learning outcomes. 
This project implements a regression model that uses chat data from Supabase and manually scored target metrics (e.g., comprehension, participation, problem-solving, etc.) to predict conversation quality.

The pipeline includes:
- **Data Loading:** Querying chat messages from Supabase and reading target scores from a CSV file.
- **Preprocessing:** Adding role tokens (`<BOT>` and `<USER>`), grouping messages by thread, and tokenizing conversation text using the Longformer tokenizer.
- **Model Training:** Training a regression model (built on top of a Longformer encoder) with multiple outputs.
- **Inference:** Using the pre-trained model to predict target scores for a new conversation.

## Features

- Efficient data loading from Supabase (only retrieving conversations with valid thread IDs taken from scores.csv).
- Comprehensive preprocessing that preserves target score columns.
- A multi-output regression model using a pre-trained Longformer.
- Training and inference pipelines with additional evaluation metrics (MSE, MAE, R²).
- Optional plotting of predicted vs. ground truth values during training.

## Directory Structure

- Please make a file called config.py in the project's root directory and save the supabase url and service key here
- all executable files are in the src folder
- Data for scores in stored in data/ in the root of the project
- Saved models are scored in the models/ directory

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/ConversationLearning.git
   cd ConversationLearning
   ```
2. **Set up a virtual environment

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
3. **Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, run the training script from the project root:

```bash
python -m src.train --plot
```
The --plot flag is optional, and will help you visualize the predicted vs ground-truth values

## Inference
After training, please run
```bash
python -m src.main
```
This will prompt you to enter a thread id of a conversation.
Please enter the thread id and let the program do its thing!

