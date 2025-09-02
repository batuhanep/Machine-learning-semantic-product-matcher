# Custom Product Matching with Fine-Tuned Sentence-Transformers

This project provides an end-to-end solution for matching product names by fine-tuning a `sentence-transformers` model on your own domain-specific data. This approach significantly improves matching accuracy compared to using general-purpose, pre-trained models, as the model learns the specific naming conventions, abbreviations, and patterns present in your product data.

The final matching score is a hybrid, combining the semantic similarity from the fine-tuned model with lexical similarity from TF-IDF and a rule-based bonus/penalty system for product volume and packaging.

## Workflow

The process is divided into two main stages:

1.  **Training:** The `train_model.py` script takes a list of known good matches (`training_matches.xlsx`) and fine-tunes a base `sentence-transformers` model. The resulting custom model is saved to the `fine_tuned_model/` directory.
2.  **Matching:** The `run_matching_with_custom_model.py` script loads this custom model and uses it to match products from a source list to a target list (`ad.xlsx`). It generates a final ranked list of matches in `adfinal_custom_model.xlsx`.

## File Structure

```
.
├── train_model.py                  # Script to train and save the custom model
├── run_matching_with_custom_model.py # Script to perform matching using the custom model
├── training_matches.xlsx           # Your input data with known good matches
├── ad.xlsx                         # Your main data file with products to be matched
└── README.md                       # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install the required Python libraries:**
    A GPU is highly recommended for the training step, but it will work on a CPU as well.
    ```bash
    pip install pandas openpyxl sentence-transformers scikit-learn numpy torch tqdm
    ```

## Usage

### Step 1: Prepare Your Data

You need to create two Excel files:

1.  **`training_matches.xlsx`**: This file is crucial for teaching the model. It must contain pairs of product names that you consider to be a correct match. The file must have two columns:
    *   `product_main`: The source product name.
    *   `product_matched`: The corresponding matched product name.

    | product_main                  | product_matched             |
    | ----------------------------- | --------------------------- |
    | Coca Cola Zero Sugar 1.5L     | COCACOLA ZERO 1,5 LT        |
    | Ülker Chocolate Wafer 5-pack  | ULKER CHOC. WAFER 5X40G     |
    | Pınar Milk 1 Litre            | PINAR SUT 1LT HALF FAT      |

2.  **`ad.xlsx`**: This is the main file containing the products you want to match. It should have at least the following columns used by the script: `productmain`, `productmatch`, and `productcode`.

### Step 2: Train the Custom Model

Run the training script from your terminal. This will read `training_matches.xlsx`, fine-tune the model, and save the output.

```bash
python train_model.py
```

This process may take some time depending on the size of your training data and your hardware (GPU vs. CPU). Upon completion, a new directory named `fine_tuned_model` will be created. This directory contains your specialized model.

### Step 3: Run the Matching Process

Once the model is trained, you can run the main matching script.

```bash
python run_matching_with_custom_model.py
```

This script will:
1.  Load your custom model from the `fine_tuned_model` directory.
2.  Process the `ad.xlsx` file.
3.  Calculate hybrid similarity scores.
4.  Generate an output file named `adfinal_custom_model.xlsx` with the best match for each product and its similarity score.

## Customization

*   **Base Model:** You can change the base model for fine-tuning by modifying the `BASE_MODEL` variable in `train_model.py`. You can find other models on the [Hugging Face Hub](https://huggingface.co/models?library=sentence-transformers).
*   **Scoring Weights:** You can adjust the weights of the SBERT and TF-IDF scores in the `hybrid_scores` calculation inside `run_matching_with_custom_model.py` to better suit your data. For example: `hybrid_scores = 0.8 * sbert_sims + 0.2 * tfidf_sims`.