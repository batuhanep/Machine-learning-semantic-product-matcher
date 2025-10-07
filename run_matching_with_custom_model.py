import pandas as pd
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def preprocess(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_volume(text):
    text = text.lower()
    patterns = [
        r'(\d+)\s*x\s*(\d+(?:[\.,]\d+)?)\s*(ml|l|g|kg|oz|lb|fl oz|pcs|pieces|rolls|pack|unit)',
        r'(\d+(?:[\.,]\d+)?)\s*(ml|l|g|kg|oz|lb|fl oz|pcs|pieces|rolls|pack|unit)',
        r'(\d+(?:[\.,]\d+)?)\s*(ml|l|g|kg|oz|lb|fl oz|pcs|pieces|rolls|pack|unit)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            if len(match.groups()) == 3: # Multi-pack case
                quantity = float(match.group(1))
                value = float(match.group(2).replace(',', '.'))
                unit = match.group(3)
                value = quantity * value
            else: # Standard case
                value = float(match.group(1).replace(',', '.'))
                unit = match.group(2)
            # Unit conversions
            if unit in ['l', 'lt']: return value * 1000
            elif unit == 'kg': return value * 1000
            elif unit == 'oz': return value * 28.35
            elif unit == 'fl oz': return value * 29.57
            elif unit == 'lb': return value * 453.59
            else: return value
    return None

#input
print("Loading main data for matching ('ad.xlsx')...")
df = pd.read_excel('ad.xlsx')
df['clean_productmain'] = df['productmain'].apply(preprocess)
df['clean_productmatch'] = df['productmatch'].apply(preprocess)
df['volume_productmain'] = df['clean_productmain'].apply(extract_volume)
df['volume_productmatch'] = df['clean_productmatch'].apply(extract_volume)


#fine tunned model
MODEL_PATH = './fine_tuned_model'
print(f"Loading our custom fine-tuned model from: {MODEL_PATH}")
try:
    model = SentenceTransformer(MODEL_PATH)
except Exception as e:
    print(f"ERROR: Could not load the model. Make sure the '{MODEL_PATH}' directory exists and you have run the 'train_model.py' script.")
    print(f"Original Error: {e}")
    exit()

# 3. create embeddngs
print("Encoding source products (with our custom model)...")
source_sbert_embeddings = model.encode(df['clean_productmain'].tolist(), show_progress_bar=True, batch_size=32)
print("Encoding target products (with our custom model)...")
target_sbert_embeddings = model.encode(df['clean_productmatch'].tolist(), show_progress_bar=True, batch_size=32)

# 4. TF-IDF 
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
all_texts = df['clean_productmain'].tolist() + df['clean_productmatch'].tolist()
vectorizer.fit(all_texts)
source_tfidf = vectorizer.transform(df['clean_productmain'])
target_tfidf = vectorizer.transform(df['clean_productmatch'])

# 5. similarities
print("Calculating SBERT similarities...")
sbert_similarities = cosine_similarity(source_sbert_embeddings, target_sbert_embeddings)
print("Calculating TF-IDF similarities...")
tfidf_similarities = cosine_similarity(source_tfidf, target_tfidf)

results = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Performing matching"):
    source_volume = row['volume_productmain']
    sbert_sims = sbert_similarities[i]
    tfidf_sims = tfidf_similarities[i]
    
    # Hybrid scoring 
    hybrid_scores = 0.7 * sbert_sims + 0.3 * tfidf_sims
    best_idx = hybrid_scores.argmax()
    best_score = hybrid_scores[best_idx]
    
    target_volume = df.iloc[best_idx]['volume_productmatch']
    
    # Volume bonus/penalty 
    volume_bonus = 0 
    if source_volume and target_volume:
        source_text = str(row['productmain']) if pd.notna(row['productmain']) else ""
        target_text = str(df.iloc[best_idx]['productmatch']) if pd.notna(df.iloc[best_idx]['productmatch']) else ""
        source_pack_match = re.search(r'(\d+)\s*x', source_text.lower())
        target_pack_match = re.search(r'(\d+)\s*x', target_text.lower())
        source_pack_count = int(source_pack_match.group(1)) if source_pack_match else 1
        target_pack_count = int(target_pack_match.group(1)) if target_pack_match else 1
        source_unit_volume = source_volume / source_pack_count
        target_unit_volume = target_volume / target_pack_count
        # Use max(..., 1) to avoid division by zero
        unit_volume_diff = abs(source_unit_volume - target_unit_volume) / max(source_unit_volume, target_unit_volume, 1)
        if source_pack_count == target_pack_count and unit_volume_diff < 0.2:
            volume_bonus = 0.15 # Same pack count + similar volume
        elif source_pack_count != target_pack_count:
            volume_bonus = -0.2 # Penalty for different pack count
        elif unit_volume_diff > 0.5:
            volume_bonus = -0.3 # Penalty for very different volume
            
    final_score = min(best_score + volume_bonus, 1.0)
    
    results.append({
        'productmain': row['productmain'],
        'matched_product': df.iloc[best_idx]['productmatch'],
        'productcode': df.iloc[best_idx]['productcode'],
        'similarity_score': f"{round(final_score * 100, 2)}%"
    })

match_df = pd.DataFrame(results)
match_df.to_excel('adfinal_custom_model.xlsx', index=False)
print("Process complete! Results saved to 'adfinal_custom_model.xlsx'.")
