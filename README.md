# FARM_tutorial

[FARM(Framework for Adapting Representation Models)](https://github.com/deepset-ai/FARM) tutorial for new users

# Blog Article

You can also see it from Blog: [simonjisu.github.io - Farm tutorial](https://simonjisu.github.io/nlp/2021/03/30/farm.html)

# Task 1: Fine-Tuning KcBert with NSMC dataset

- KcBert: [https://github.com/Beomi/KcBERT](https://github.com/Beomi/KcBERT)
- NSMC dataset: [https://github.com/e9t/nsmc](https://github.com/e9t/nsmc)

## Type 1. Tutorial in Colab (Korean)

Just follow(`Shift` + `Enter`) it! `FARM_Tutorial.ipynb`: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simonjisu/FARM_tutorial/blob/main/notebooks/FARM_colab.ipynb)

## Type 2. Fork the project and run following code

Please `fork` the github to your repository, then change the setting of `./run_text_classification.sh`

```
git clone https://github.com/[username]/FARM_tutorial.git
cd FARM_tutorial
conda create --name farm python=3.7 -y
pip install requirements.txt
git clone https://github.com/e9t/nsmc
python ./src/preprocess.py
sh ./run_text_classification.sh
```

# Task 2: Fine-Tuning KoELECTRA with AIhub QA dataset

## Type 1. Tutorial in Colab

> Download the pretrained model and see the results

Just follow(`Shift` + `Enter`) it! `FARM_Tutorial_QA.ipynb`: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simonjisu/FARM_tutorial/blob/main/notebooks/FARM_colab_QA.ipynb)

## Type 2. Fork the project and run following code

Please `fork` the github to your repository, then change the setting of `./run_qa.sh`

```
git clone https://github.com/[username]/FARM_tutorial.git
cd FARM_tutorial
conda create --name farm python=3.7 -y
pip install requirements.txt
git clone https://github.com/e9t/nsmc
python ./src/preprocess.py
sh ./run_qa.sh
```

# To Be Added

- [x] Fine-Tuning with QA dataset with docker