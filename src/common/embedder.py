import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import diskcache
from common.config import EmbModel, get_device
from common.data import TextDataset

CACHE_DIR_BASE = "cache/"

logger = logging.getLogger(__name__)


class EmbeddingService:
    @dataclass
    class Params:
        model_name: EmbModel
        max_length: int = 64
        batch_size: int = 256

    """
    Encapsulates a "frozen" text embedding model, its tokenizer,
    and a persistent DISK cache for efficient, deduplicated embedding.
    Now includes an in-memory cache layer for speed.
    """

    @staticmethod
    def create(params: Params):
        return EmbeddingService(**params.__dict__)

    def __init__(self, model_name: str | EmbModel, max_length: int, batch_size: int):
        logger.info(f"Loading embedding model: {model_name}...")
        model_name = str(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length

        cache_key = f"{model_name.replace('/', '__')}" #_len{max_length}
        cache_dir = os.path.join(CACHE_DIR_BASE, cache_key)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.embedding_cache = diskcache.Cache(cache_dir)

        self.memory_cache: dict[str, np.ndarray] = {}

        logger.info(f"Model {model_name} loaded onto {self.device}. Cache at {cache_dir}")

    def embed(self, text_list: list[str]) -> np.ndarray:
        """
        Embeds a list of texts.
        Priority: Memory Cache -> Disk Cache -> Compute
        """
        if not text_list:
            return np.array([])

        unique_texts = set(text_list)
        texts_to_compute = []

        # 1. Check caches (Memory -> Disk)
        for text in unique_texts:
            # If already in memory, good.
            if text in self.memory_cache:
                continue

            # If on disk, load to memory.
            if text in self.embedding_cache:
                self.memory_cache[text] = self.embedding_cache[text]
            else:
                # Not in either, needs computation
                texts_to_compute.append(text)

        logger.info(f'{len(text_list)=} {len(unique_texts)=} {len(texts_to_compute)=} (to compute)')

        # 2. Embed *only* the new texts (if any)
        if texts_to_compute:
            self._compute_and_cache(texts_to_compute)

        # 3. Map all texts back to the original list order using ONLY memory
        # This is now fast because it's a pure dict lookup
        final_embeddings = [self.memory_cache[text] for text in text_list]

        return np.stack(final_embeddings)

    def _compute_and_cache(self, texts: list[str]):
        """
        Helper to compute embeddings for a list of texts and save them
        to both disk and memory caches.
        """
        total_samples = len(texts)
        num_batches = ((total_samples - 1) // self.batch_size) + 1
        logger.info(f"  Embedding {total_samples} new unique texts in {num_batches} batches...")

        dataset = TextDataset(texts)
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

            # Move to CPU immediately to free GPU
            batch_embeddings_np = embeddings.cpu().numpy()
            all_new_embeddings.append(batch_embeddings_np)

            processed_samples += len(batch_texts)

            # Optional: Log progress
            if num_batches > 10 and processed_samples % (self.batch_size * 5) == 0:
                # elapsed = time.time() - start_time
                logger.info(f"    Processed {processed_samples}/{total_samples}...")

        elapsed_time = time.time() - start_time
        logger.info(f"  Computation complete. Total time: {elapsed_time:.2f}s")

        if all_new_embeddings:
            # Flatten list of arrays
            all_new_embeddings_np = np.concatenate(all_new_embeddings, axis=0)

            # Update both caches
            for text, embedding in zip(texts, all_new_embeddings_np):
                self.embedding_cache[text] = embedding  # Persist to Disk
                self.memory_cache[text] = embedding  # Keep in RAM
        else:
            logger.warning("  texts_to_compute was not empty, but no embeddings were generated.")
