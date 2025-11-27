# **Multi: Recurring Transaction Pattern Detector**

**Multi** is a deep learning system designed to identify, group, and analyze recurring patterns in bank transaction histories (e.g., Subscriptions, Salary, Rent, Utilities).

Unlike traditional classifiers that look at one transaction at a time, Multi uses a **Set-to-Graph Transformer** architecture to analyze the entire history of an account simultaneously, allowing it to learn complex contextual relationships between transactions.

## **🚀 Key Features**

* **Set-to-Graph Architecture:** Contextualizes every transaction against every other transaction in the account history.  
* **Frozen Embedding Integration:** Uses a decoupled EmbeddingService (BERT/MiniLM) for high-performance text encoding without re-computation overhead.  
* **Dual-Head Prediction:** Simultaneously predicts:  
  1. **Linkage (Clustering):** Which transactions belong to the same group?  
  2. **Cycle (Classification):** Is this group Monthly, Weekly, or Non-recurring?  
* **A100 Optimized:** Implements Dynamic Padding and Credit/Debit splitting to maximize GPU throughput.

## **🧠 Model Architecture**

The model processes an account's history (set of transactions) and outputs a graph structure where nodes are transactions and edges represent "belonging to the same pattern."

### **1\. The Inputs**

Each transaction is converted into a fused vector:

* **Text:** Encoded via frozen LLM (e.g., all-MiniLM-L6-v2) $\\rightarrow$ Projected to hidden dim.  
* **Amount:** Log-normalized ($\\log(1+|x|)$) and sign-encoded.  
* **Time:** Days since the first transaction (Positional encoding).

### **2\. The Contextualizer (Transformer)**

A standard **Transformer Encoder** attends to all transactions. This allows the model to understand that two transactions are related not just because they look similar, but because they fit a temporal rhythm.

### **3\. The Outputs**

* **Adjacency Head (Bilinear):** Outputs an $N \\times N$ matrix. Cell $(i, j)$ is the probability that transaction $i$ and $j$ are the same pattern.  
* **Cycle Head (Linear):** Outputs the probability distribution (Monthly, Weekly, None) for each transaction.

## **📂 Package Structure**

multi
├── config.py       \# Hyperparameters (Batch size, Dimensions, Labels)  
├── data.py         \# Data Loader, Dynamic Padding, Credit/Debit Splitting  
├── encoder.py      \# TransactionTransformer, Encoder, Heads  
├── train.py        \# Training Loop & Loss Calculation  
├── inference.py    \# Graph Clustering & Pattern Extraction  
├── eval\_multi.py   \# Advanced Metrics (B-CUBED, PR-AUC)  
├── run\_multi.py    \# CLI Entry point for Training  
└── README.md       \# This file

## **🏃‍♂️ Usage**

### **1\. Training**

The training script automatically splits accounts into Training/Validation sets.

\# Run with real data  
python multi/run\_multi.py \--data /path/to/transactions.csv \--epochs 10 \--batch\_size 32

\# Run with synthetic mock data (for testing pipeline)  
python multi/run\_multi.py \--epochs 5

**Key Flags:**

* \--model\_name: The HuggingFace model for embeddings (default: sentence-transformers/all-MiniLM-L6-v2).  
* \--epochs: Number of training passes.

### **2\. Evaluation**

To run the full suite of metrics (B-CUBED, PR-AUC, etc.) on a test set:

python multi/src/eval\_multi.py \--data /path/to/test.csv \--model checkpoints/model.pth

### **3\. Inference (Production)**

To use the model in a production pipeline to label new data:

from multi.config import MultiExpConfig  
from multi.inference import MultiPredictor  
from common.embedder import EmbeddingService

\# 1\. Setup  
config \= MultiExpConfig()  
emb\_params \= EmbeddingService.Params(model\_name="sentence-transformers/all-MiniLM-L6-v2")  
service \= EmbeddingService.create(emb\_params)

\# 2\. Load Model  
predictor \= MultiPredictor("checkpoints/model.pth", config, service)

\# 3\. Predict  
\# Returns DataFrame with 'pred\_patternId', 'pred\_patternCycle', 'pred\_recurring\_prob'  
results\_df \= predictor.predict(new\_data\_df)

## **📊 Evaluation Metrics**

We use a **Cascade Evaluation** strategy to ensure quality at every level:

| Level | Metric | Description |
| :---- | :---- | :---- |
| **1\. Detection** | **F1 Score** & **PR-AUC** | Can the model distinguish recurring transactions from one-off noise? PR-AUC is critical due to class imbalance (\~20% recurring). |
| **2\. Grouping** | **B-CUBED F1** | The gold standard for clustering. Penalizes over-segmentation (splitting Netflix into two groups) and over-merging. |
| **3\. Cycle** | **Accuracy (on TP)** | If the model correctly found a pattern, did it correctly label it "Monthly" vs "Weekly"? |

## **🔧 Data Format**

Input CSV should contain at least:

* accountId: Unique user ID.  
* date: Transaction date.  
* amount: Transaction amount.  
* bankRawDescription: Description text.  
* counter\_party: Normalized merchant name (optional, can be empty).  
* isRecurring: (Bool) Target for training.  
* patternId: (String) Ground truth cluster ID.  
* patternCycle: (String) Ground truth cycle label.