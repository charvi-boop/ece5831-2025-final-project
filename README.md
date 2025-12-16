# ece5831-2025-final-project

# Intelligent Ticket Triage (ECE 5831 Final Project)

**Automatic Customer Support Classification using Fine-Tuned Llama 3**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%26%20Streamlit-red)
![Model](https://img.shields.io/badge/AI-Llama%203%208B-green)

## Team

| Name |
| :--- |
| **Shreyas Katale** |
| **Nikhil Patil** |
| **Charvi Rathod** |

## Project Overview
In the modern service economy, efficient handling of customer support inquiries is critical. Manual triage of support tickets is slow, expensive, and prone to human error.

This project automates the classification of customer complaints into **20 distinct financial product categories** (e.g., Mortgage, Credit Card, Fraud) using Natural Language Processing. We benchmarked two approaches:
1.  **Statistical Baseline:** TF-IDF + Logistic Regression.
2.  **Generative AI Solution:** A **Meta-Llama-3-8B** model fine-tuned using **QLoRA** (Quantized Low-Rank Adaptation) on a single GPU.

Our fine-tuned LLM achieved a **Weighted F1-Score of 0.7333**, matching the baseline's accuracy while demonstrating superior data efficiency (requiring only ~20% of the training data).

---
## Results & Performance

We evaluated both the Baseline and the Fine-Tuned LLM on a held-out test set of **3,000 samples**.

### Head-to-Head Comparison
| Metric | Baseline (TF-IDF) | Llama 3 (QLoRA) |
| :--- | :--- | :--- |
| **Accuracy** | 0.7300 | **0.7400** |
| **Weighted F1-Score** | 0.7300 | **0.7333** |
| **Training Efficiency** | Required 100% of Dataset | **Achieved with ~20% of Dataset** |

### 🔍 Detailed Class Performance
The Llama 3 model achieved near-perfect accuracy on distinct financial products but faced challenges with overlapping legacy categories.

| Top Performing Classes | F1-Score |
| :--- | :--- |
| **Mortgage** | 0.90 |
| **Money transfers** | 0.89 |
| **Student loan** | 0.80 |

<details>
<summary><strong>Click here to see the full Classification Report</strong></summary>

| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Bank account or service | 0.67 | 0.17 | 0.27 | 12 |
| Checking or savings account | 0.79 | 0.83 | 0.81 | 139 |
| Credit card | 0.31 | 0.24 | 0.27 | 79 |
| Credit card or prepaid card | 0.48 | 0.56 | 0.52 | 91 |
| Credit reporting, repair, etc. | 0.64 | 0.73 | 0.69 | 706 |
| Credit reporting or other... | 0.82 | 0.80 | 0.81 | 1268 |
| Debt collection | 0.75 | 0.72 | 0.73 | 335 |
| Money transfer, virtual currency... | 0.90 | 0.89 | 0.89 | 100 |
| Mortgage | 0.89 | 0.90 | 0.90 | 110 |
| Student loan | 0.76 | 0.84 | 0.80 | 50 |
| Vehicle loan or lease | 0.62 | 0.43 | 0.51 | 35 |
| *... (Minority classes omitted)* | | | | |
| **TOTAL ACCURACY** | | | **0.74** | **3000** |
| **WEIGHTED AVG** | **0.73** | **0.74** | **0.73** | **3000** |

</details>

## Project Resources & Links

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Dataset & Report** | Google Drive folder containing the CSV dataset, PDF Report, and Slides. | [[https://drive.google.com/drive/folders/1KGULt9nv4bG53hxATVYuTiYqbf0Xxg1i?usp=drive_link](https://drive.google.com/drive/folders/1TNmuNnxwFxymEcARDIVHfYextvsY_HE2)] |
| **Demo Video** | YouTube video demonstrating the Streamlit App in action. | [https://youtu.be/neRFyvbH4SM] |
| **Presentation** | Recorded video presentation of the project slides. | [https://drive.google.com/file/d/1cYzX__Y4e_97MptQ_-8OUx4kb-JF0onY/view?usp=sharing] |
| **Final Report** | PDF detailed report of methodology and results. | [ECE-5831-Final-Report-Intelligent-Traige-For-Customer-Support-Tickets] |

---
