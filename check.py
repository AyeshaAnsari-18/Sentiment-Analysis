import pandas as pd

# Load the large CSV file in chunks
chunk_size = 500000
chunks = pd.read_csv("data/ratings_Electronics.csv", chunksize=chunk_size)

# Save each chunk as a separate file
for i, chunk in enumerate(chunks):
    chunk_filename = f"ratings_Electronics_part_{i+1}.csv"
    chunk.to_csv(chunk_filename, index=False)
    print(f"Saved {chunk_filename}")
