# bert-morph-tagger-udmurt

This repository contains the training code for an encoder-based (BERT-like) model for morphological tagging. The trained model performs fine-grained token-level morphological parsing.

🤗 [Morphological parser for Udmurt on HuggingFace Hub](https://huggingface.co/ulyanaisaeva/bert-morph-tagger-udmurt) 🤗

The training code was used to train a morhological tagging model for the Udmurt language, which is morphologically rich and relatively low-resource. The model is based on [`cis-lmu/glot500-base`](https://huggingface.co/cis-lmu/glot500-base) multilingual encoder-only model fine-tuned for morphological analysis. The model weights are available on the Hugging Face Hub.


## Features

- Supports both standart cross-entropy training and soft training for morhological ambiguity
- Training implemented using the `transformers` library for seamless compatibility with Hugging Face ecosystem
- Easily adaptable to other languages

**NB: the [training data](http://udmurt.web-corpora.net/) used in this project is not a public dataset but is available for researchers by request to the owner.**


## Command-line Training

The repository provides two training scripts: `pretrain.py` for pretraining with multilabel loss and `finetune.py` for standard fine-tuning.

### Data Format

Training and validation data should be provided as JSON files with the following structure:

```json
[
  {
    "words": ["шуэм", "аслыз", ":"],
    "labels": [["VERB,Evident=Fh|Number=Sing|Person=3"], ["PRON,Case=Dat|Number[psor]=Sing|Person[psor]=3"], ["PUNCT"]],
    "labels_amb": [["VERB,Evident=Fh|Number=Sing|Person=3"], ["PRON,Case=Dat|Number[psor]=Sing|Person[psor]=3"], ["PUNCT"]]
  }
]
```

Each entry contains:
- `words`: list of tokens (words)
- `labels`: list of label lists (for multilabel) or single labels (for single-label training)
- `labels_amb`: (optional) list of ambiguous label variants for evaluation metrics

### Configuration

All parameters can be configured via YAML config files. Each script has its own default config:

- `config/finetune_config.yaml` - default config for fine-tuning
- `config/pretrain_config.yaml` - default config for pretraining

Each config file has the following sections:

- `paths`: Paths to tokenizer, models, and data files
- `experiment`: Experiment settings (name, train size limit for finetune)
- `training`: Training hyperparameters (learning rate, batch sizes, epochs, etc.)

All parameters must be specified in the config file. You can specify a custom config file using the `--config` argument.

### Pretraining

Run pretraining with multilabel loss:

```bash
python pretrain.py
```

### Fine-tuning

Run fine-tuning with standard cross-entropy loss:

```bash
python finetune.py
```

**Note:** The hyperparameters in the default config are example values and should be adjusted based on dataset size, hardware capabilities, and convergence behavior. The default values may not reproduce the published results without proper tuning.


## Citation

If you use this repository or the model in your research, please cite the related publication:

```bibtex
@inproceedings{isaeva-etal-2025-combining,
    title = "Combining Automated and Manual Data for Effective Downstream Fine-Tuning of Transformers for Low-Resource Language Applications",
    author = "Isaeva, Ulyana  and
      Astafurov, Danil  and
      Martynov, Nikita",
    booktitle = "Proceedings of the 1st Joint Workshop on Large Language Models and Structure Modeling (XLLM 2025)",
    month = aug,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.xllm-1.9/",
    doi = "10.18653/v1/2025.xllm-1.9",
    pages = "86--90"
}
```


---

Feel free to open issues or contribute improvements via pull requests.  
For questions or support, please contact ulyana.isaeva20@gmail.com.
