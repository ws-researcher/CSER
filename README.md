# CSER

## The code structure is as follows
```
├── datasets
    ├── MATRES
    ├── TimeBank-dense
├── pretrained_models
    ├── roberta-base
    ├── roberta-large
├── SourceCode
    ├── data_loader
    ├── models
    ├── utils
    ├── IE.py
    ├── main.py

> Unzip the compressed file dataset into the folder datasets
> python main.py --dataset TBD --roberta_type roberta-large --bs 16
> python main.py --dataset MATRES --roberta_type roberta-large --bs 16
