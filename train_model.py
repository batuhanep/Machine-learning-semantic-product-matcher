import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. training data
print("Loading training data...")
try:
    train_df = pd.read_excel('training_matches.xlsx')

    train_df = train_df[['product_main', 'product_matched']]
    train_df.dropna(inplace=True)
except FileNotFoundError:
    print("ERROR: 'training_matches.xlsx' not found. Please save your training data with this name.")
    exit()
except KeyError:
    print("ERROR: 'training_matches.xlsx' must contain 'product_main' and 'product_matched' columns.")
    exit()

# good match
train_examples = []
for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Preparing training examples"):
    text1 = str(row['product_main'])
    text2 = str(row['product_matched'])
    # We provide a score of 1.0 to indicate that these two texts are highly similar.
    train_examples.append(InputExample(texts=[text1, text2], label=1.0))

print(f"Created {len(train_examples)} training examples.")

# 2. Define the Model
# We choose a base model to fine-tune. A multilingual model is a good starting point.
# 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' is a balanced and effective choice.
BASE_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
print(f"Loading base model: {BASE_MODEL}")
model = SentenceTransformer(BASE_MODEL)

# 3. Specify Training Parameters
# CosineSimilarityLoss is designed to bring the embeddings of sentence pairs closer together.
# It's the most suitable loss function for our task.
train_loss = losses.CosineSimilarityLoss(model)

# The DataLoader efficiently feeds the data to the model.
# batch_size determines how many examples are processed in each step.
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Number of training epochs (how many times to iterate over the entire dataset)
epochs = 4
# Warmup steps (initial steps where the learning rate gradually increases)
warmup_steps = int(len(train_dataloader) * epochs * 0.1) # 10% of total steps

# 4. Train the Model
print("Starting model training...")
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=epochs,
          warmup_steps=warmup_steps,
          output_path='./fine_tuned_model', # Directory where the fine-tuned model will be saved
          show_progress_bar=True)

print("\nTraining complete!")
print("Model saved to the 'fine_tuned_model' directory.")
