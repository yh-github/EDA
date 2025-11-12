import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import diskcache
from config import BaseModel
from data import TextDataset

CACHE_DIR_BASE = "cache/"

logger = logging.getLogger(__name__)

class EmbeddingService:

    @dataclass
    class Params:
        model_name: BaseModel
        max_length: int = 64
        batch_size: int = 256 # Define a batch size for text embedding

    """
    Encapsulates a "frozen" text embedding model, its tokenizer,
    and a persistent DISK cache for efficient, deduplicated embedding.
    """

    @staticmethod
    def create(params:Params):
        return EmbeddingService(**params.__dict__)

    def __init__(self, model_name: str|BaseModel, max_length: int, batch_size: int):
        logger.info(f"Loading embedding model: {model_name}...")
        model_name = str(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length
        cache_dir = CACHE_DIR_BASE+model_name.replace("/", "__")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.embedding_cache = diskcache.Cache(cache_dir)
        logger.info(f"Model {model_name} loaded onto {self.device}. Cache at {cache_dir}")

    def embed(self, text_list: list[str]) -> np.ndarray:
        """
        Embeds a list of texts, using the internal disk cache and batching
        to avoid re-embedding texts it has already seen.
        """
        if not text_list:
            return np.array([])

        unique_texts = set(text_list)
        texts_to_embed = [text for text in unique_texts if text not in self.embedding_cache]

        logger.info(f'{len(text_list)=} {len(unique_texts)=} {len(texts_to_embed)=}')

        # 2. Embed *only* the new texts (if any)
        if texts_to_embed:
            total_samples = len(texts_to_embed)
            num_batches = ((total_samples - 1) // self.batch_size) + 1
            logger.info(f"  Embedding {total_samples} new unique texts in {num_batches} batches...")
            dataset = TextDataset(texts_to_embed)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            all_new_embeddings = []
            start_time = time.time()
            processed_samples = 0

            for batch_texts in dataloader:
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                if not hasattr(outputs, 'pooler_output') or outputs.pooler_output is None:
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                else:
                    embeddings = outputs.pooler_output

                all_new_embeddings.append(embeddings.cpu())

                processed_samples += len(batch_texts)
                elapsed_time = time.time() - start_time
                time_per_sample = elapsed_time / processed_samples if processed_samples > 0 else 0
                estimated_time_remaining = (total_samples - processed_samples) * time_per_sample

                # Use logger.info for progress, with \r for single-line update
                logger.info(
                    f"Processed {processed_samples}/{total_samples} samples | "
                    f"Elapsed: {elapsed_time:.2f}s | "
                    f"Est. Remaining: {estimated_time_remaining:.2f}s"
                )

            logger.info(f"Embedding complete. Total time: {elapsed_time:.2f}s")

            if all_new_embeddings:
                all_new_embeddings_np = torch.cat(all_new_embeddings).numpy()
                for text, embedding in zip(texts_to_embed, all_new_embeddings_np):
                    self.embedding_cache[text] = embedding
            else:
                logger.warning("  texts_to_embed was not empty, but no embeddings were generated.")

        # Map all texts (new and cached) back to the original list order
        # This now pulls from the disk cache
        final_embeddings = [self.embedding_cache[text] for text in text_list]

        return np.stack(final_embeddings)

    def __del__(self):
        """Ensure the cache is closed when the object is destroyed."""
        if hasattr(self, 'embedding_cache'):
            self.embedding_cache.close()