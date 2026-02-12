CHASM
Chunk-Hierarchical Abstractive Summarization Module

The system combines:
Graph-based salience estimation
Hierarchical sentence selection
Abstractive generation (BART)
Redundancy reduction
Optional fact-consistency verification
CHASM is evaluated on:
PubMed biomedical abstracts
arXiv scientific papers
This enables rigorous cross-domain generalization analysis.

🧠 Problem Statement

Long scientific documents:

Contain complex structure

Exhibit high redundancy

Include domain-specific terminology

Are difficult for flat summarization models

CHASM addresses these challenges using hierarchical chunk-based abstraction and salience fusion.

🏗 System Architecture
Raw Document
     ↓
Preprocessing & Cleaning
     ↓
Chunk Segmentation
     ↓
Graph-Based Salience Scoring
     ↓
Hierarchical Re-ranking
     ↓
Abstractive Generation (BART)
     ↓
Redundancy Reduction
     ↓
(Optional) Fact Verification

🔍 Core Modules
Module	Description

graph_builder.py	Sentence similarity graph construction

salience_model.py	Fusion of salience signals

hierarchical_ranker.py	Chunk-aware sentence selection

bart_generator.py	Abstractive summary generation

rewriter.py	Redundancy-aware rewriting

verifier.py	Fact-consistency verification

scorers.py	ROUGE, BERTScore, redundancy evaluation



🚀 Installation
1️⃣ Clone the Repository
git clone <your-repository-url>
cd Suvidha_Internship
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Install CUDA-Enabled PyTorch (RTX 3050 Recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Verify GPU:

python
>>> import torch
>>> torch.cuda.is_available()
True
▶️ Running Experiments
Run CHASM (Full Model)
python scripts/run_chasm_full_arxiv.py
Run Baselines

TextRank:

python scripts/run_textrank_arxiv.py

BART Baseline:

python scripts/run_bertsum_arxiv.py
Evaluate Model
python scripts/evaluate_arxiv_chasm.py

Metrics are saved to:

experiments/metrics/
📊 Evaluation Metrics

CHASM is evaluated using:

ROUGE-1

BERTScore (F1)

Redundancy

Example Cross-Domain Results
Dataset	ROUGE-1	BERTScore	Redundancy
PubMed	~0.48	~0.92	~0.05
arXiv	0.33	0.81	0.02
Interpretation

Moderate lexical drop across domains

Strong semantic preservation

Reduced redundancy in long documents

This demonstrates domain robustness.

🧪 Reproducibility

All experiments:

Save intermediate results

Save baseline outputs

Save metric JSON files

Support visualization via Jupyter notebooks

🖥 Hardware Used

GPU: NVIDIA RTX 3050 Laptop GPU

CUDA: 11.8

RAM: 16GB

OS: Windows

🔬 Research Contributions

Hierarchical chunk-based abstraction

Graph-salience fusion mechanism

Redundancy-aware generation

Optional factual verification

Cross-domain validation (biomedical → scientific)

📈 Future Work

Hallucination detection module
Longformer integration
Domain-adaptive fine-tuning

Structured medical fact alignment

📜 Citation
@article{chasm2026,
  title={CHASM: Chunk-Hierarchical Abstractive Summarization for Scientific Documents},
  author={Chauhan, Rosesh},
  year={2026}
}

📌 License

This project is released for academic and research purposes.