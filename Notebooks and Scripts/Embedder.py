import asyncio

import numpy as np
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from loguru import logger

# Configure loguru for console logging only
logger.remove()
logger.add(sink=lambda message: print(message, flush=True), colorize=True, level="INFO")

# File paths
input_csv = r"S:\PLP-Project\Output\Chunked_Full.csv"
output_csv = r"S:\PLP-Project\Output\embedded_chunks.csv"
model_path = r"S:\PLP-Project\models\gte-embedding"

model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

async def embed_chunks(df, batch_size=100):
    logger.info("Starting embedding process")
    all_embeddings = []
    
    for i in range(0, len(df), batch_size):
        batch = df['chunk_text'][i:i+batch_size].tolist()
        logger.info(f"Processing batch {i//batch_size + 1}/{len(df)//batch_size + 1}")
        
        try:
            batch_embeddings = await embeddings.aembed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
            all_embeddings.extend([None] * len(batch))
    
    return all_embeddings

async def main():
    logger.info(f"Loading chunks from {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} chunks")

    embeddings_list = await embed_chunks(df)

    logger.info("Adding embeddings to DataFrame")
    df['embedding'] = embeddings_list

    logger.info(f"Saving results to {output_csv}")
    df.to_csv(output_csv, index=False)
    logger.success(f"Embedded chunks saved to {output_csv}")

if __name__ == "__main__":
    asyncio.run(main())
