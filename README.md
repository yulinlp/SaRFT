# SaRFT

**SaRFT: Safety-aware Role-play Fine-Tuning**

This is the official code repository for our ACL 2025 paper:

> **"Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs"**

## 🌟 Overview

Role-playing enables large language models (LLMs) to deliver immersive and personalized user experiences. However, it also introduces significant safety risks when models are fine-tuned to adopt harmful or unethical personas. Despite advances in role-play fine-tuning techniques that enhance role adaptability, many of these methods inadvertently compromise model safety—especially when training for antagonistic or morally ambiguous characters.

This project introduces **SaRFT (Safety-Aware Role-Play Fine-Tuning)**, a novel framework designed to:

- ⚖️ **Systematically assess safety risks** associated with role-play fine-tuning  
- 🛡️ **Mitigate unsafe behaviors** while preserving strong role-adaptation performance  
- 📊 **Provide standardized evaluation protocols and datasets** for measuring both safety and role-play fidelity

## 📁 Project Structure

```bash
SaRFT/
├── decoding/             # Decoding-related scripts and code
│   ├── scripts/          # Decoding scripts
│   └── src/              # Source code for decoding
├── evaluation/           # Evaluation module
│   ├── datasets/         # Datasets used for evaluation
│   ├── metrics/          # Metric calculation code
│   ├── scripts/          # Evaluation scripts
│   ├── run.py            # Main evaluation script
│   └── utils.py          # Utility functions for evaluation
├── sarft/                # Core model code and configurations
│   ├── config/           # Model configuration files
│   ├── ds_configs/       # Dataset-related configuration files
│   ├── scripts/          # Training/inference scripts
│   └── src/              # Main source code for the SaRFT model
└── requirements.txt      # Dependency list
````

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yulinlp/SaRFT.git
cd SaRFT
```

### 2. Install dependencies

We recommend using a virtual environment:

```bash
pip install -r requirements.txt
```

### 3. Run training or evaluation

Example:

```bash
python sarft/scripts/run.sh 0.5 dpo Jigsaw
python evaluation/scripts/eval.sh SaRFT_Jigsaw
```

## 📊 Datasets

We use several safety-focused datasets to analyze role-play behavior. Please refer to our paper for detailed descriptions and download links.

After downloading, organize the datasets following the structure expected by the `data/` directory in this repository.

## 📌 Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{zhao2025beware,
  title={Beware of your po! measuring and mitigating ai safety risks in role-play fine-tuning of llms},
  author={Zhao, Weixiang and Hu, Yulin and Deng, Yang and Guo, Jiahe and Sui, Xingyu and Han, Xinyang and Zhang, An and Zhao, Yanyan and Qin, Bing and Chua, Tat-Seng and others},
  journal={arXiv preprint arXiv:2502.20968},
  year={2025}
}
```

## 🛠️ License

This project is licensed under the MIT License.

## 🤝 Contributions

We welcome contributions! Please open an issue or pull request if you have ideas or bugfixes to share.
