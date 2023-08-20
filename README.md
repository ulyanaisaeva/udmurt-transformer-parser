# Udmurt-transformer-parser

Transformer-based POS-tagging model for Udmurt. Based on the `bert-base-multilingual-uncased` pretrained on Udmurt texts. The pretrained BERT is published on the [HuggingFace Hub](https://huggingface.co/ulyanaisaeva/udmurt-bert-base-uncased).


**NB: the [training data](http://udmurt.web-corpora.net/) used in this project is not a public dataset but is available for researchers by request to the owner. The following code is organized based on an assumption that the raw files are in the `data_raw` folder and after preprocessing the are in the `data` folder.**

The model is temporarily stored on Google Drive. You may download it [here](https://drive.google.com/file/d/10zolxyFu52JI78kyVVRY0ub7SJ4BXZtf/view) or via command line (requires `gdown`):

```bash
gdown -O "models/udm_morph_tagger.model" "10zolxyFu52JI78kyVVRY0ub7SJ4BXZtf" 
```

- BERT pretraining: [Extending_BERT.ipynb](https://github.com/ulyanaisaeva/udmurt-transformer-parser/blob/main/Extending_BERT.ipynb)
- POS-tagger training: [Train_POStagger.ipynb](https://github.com/ulyanaisaeva/udmurt-transformer-parser/blob/main/Train_POStagger.ipynb)


<details><summary>Repo structure</summary>
<p>

```
.
├── data        <-- files in .txt and .conllu
│   ├── conllu
│   │   └── example.conllu
│   └── txt
│       └── example.txt
├── data_raw    <-- original files
├── model_utils     <-- tagger class and data utils
│   ├── modify_model.py
│   ├── transformer_tagger.py
│   └── ud_dataset_utils.py
├── models                  <-- trained models are saved here
├── tokenizer               <-- trained tokenizers are saved here
│   └── match_vocabs.py
├── Extending_Bert.ipynb    <-- notebook to pretrain BERT on `data`
├── Train_POStagger.ipynb   <-- notebook to train a POS-tagger
```

</p>
</details>


### Usage (requires GPU)

```python
import pickle
from model_utils.transformer_tagger import *

with open("models/udm_morph_tagger.model", "rb") as model_file:
    model = pickle.load(model_file)

test_data_file = "data/conllu/example.conllu"
test_dataset = UDDataset(read_infile(test_data_file), tokenizer=tokenizer)

model.predict(test_dataset)
```
