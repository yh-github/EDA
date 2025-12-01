Multi Model Architecture

The Multi model is a Set-to-Graph Transformer designed to detect recurring patterns within a set of bank transactions. Unlike sequential models (RNNs) that view transactions as a time series, Multi treats an account's history as a set of nodes, allowing it to learn complex, non-sequential dependencies (e.g., a monthly subscription appearing on irregular dates).

1. Input Features

The model accepts a variable-length sequence of transactions (an account history). Each transaction is treated as a node with the following features:

A. Text Features (Description)

Source: bankRawDescription (and optionally counter_party).

Preprocessing: Tokenized using a pre-trained HuggingFace tokenizer (e.g., sentence-transformers/all-mpnet-base-v2).

Encoding: Passed through a frozen pre-trained Language Model (PLM). The pooled output (mean pooling of the last hidden state) is projected to the model's hidden_dim.

B. Amount Features

Source: amount.

Preprocessing: Log-normalized to handle the varying magnitude of financial data while preserving direction (credit/debit).


$$x_{amt} = \text{sign}(amount) \times \log(1 + |amount|)$$

Encoding: Projected via a Linear layer to hidden_dim.

C. Relative Time (Frequency Signal)

Source: date.

Preprocessing: Calculated as the number of days elapsed since the first transaction in the history.

Encoding: A Sinusoidal Time Encoding (similar to Positional Encoding in vanilla Transformers) is used to allow the model to learn periodicities (e.g., "every 30 days", "every 7 days").


$$PE(pos, 2i) = \sin(pos / 10000^{2i/d_{model}})$$

D. Calendar Features (Phase Signal)

Source: date.

Preprocessing: Explicit cyclic features to capture "phase" (e.g., "happens on the 1st of the month" or "happens on Mondays").

sin(day_of_week), cos(day_of_week)

sin(day_of_month), cos(day_of_month)

Encoding: Concatenated and projected via a Linear layer to hidden_dim.

2. The Encoder (Feature Fusion)

For every transaction $i$ in the input set, the feature embeddings are summed to create a single fused node embedding $x_i$:

$$x_i = \text{LayerNorm}(\text{Proj}(Txt_i) + \text{Proj}(Amt_i) + \text{TimeEnc}(Day_i) + \text{Proj}(Cal_i))$$

3. The Contextualizer (Transformer Backbone)

The core of the model is a standard Transformer Encoder (stack of TransformerEncoderLayers).

Input: The set of fused node embeddings $\{x_1, x_2, ..., x_N\}$.

Mechanism: Self-Attention ($O(N^2)$) allows every transaction to attend to every other transaction in the history.

Output: A contextualized embedding $h_i$ for each transaction that contains information about the transaction itself and its relationship to the rest of the history.

4. Prediction Heads

The model utilizes two distinct heads to solve the problem of detecting recurring payments.

Head A: Adjacency (Clustering)

Predicts the probability that any two transactions $i$ and $j$ belong to the same recurring pattern (linkage).

Type: Bilinear / Dot-Product Attention.

Mechanism:

Project $h$ to a query space: $h' = W_{adj} h$.

Compute similarity matrix: $A_{logits} = h' \cdot h^T$.

Enforce symmetry: $A_{final} = \frac{A_{logits} + A_{logits}^T}{2}$.

Output: An $N \times N$ matrix where cell $(i, j)$ represents the logit probability that transaction $i$ and $j$ are linked.

Loss: Binary Cross Entropy (BCE) or Focal Loss on the edge existence.

Head B: Cycle Classification

Predicts the specific recurrence cycle (e.g., "Monthly", "Weekly", "None") for each transaction.

Type: Linear Classification.

Mechanism: $C_{logits} = W_{cycle} h_i + b$.

Output: A vector of size $(N, \text{NumClasses})$ for each transaction.

Classes typically include: None (Noise), Monthly, Weekly, Bi-Weekly, etc.

Loss: Cross Entropy Loss.

5. Summary Flow

graph TD
    subgraph Inputs
    T[Text Description] --> PLM[Frozen LLM]
    A[Amount] --> Log[Log Transform]
    D[Date] --> Time[Time Encoding]
    C[Calendar] --> Cyclic[Cyclic Feats]
    end

    PLM --> Fusion((Sum & Norm))
    Log --> Fusion
    Time --> Fusion
    Cyclic --> Fusion

    Fusion --> Trans[Transformer Encoder\n(Contextualizes History)]

    Trans --> H[Contextualized Embeddings h]

    H --> Adj[Adjacency Head\n(NxN Linkage Matrix)]
    H --> Cyc[Cycle Head\n(Nx1 Classification)]
