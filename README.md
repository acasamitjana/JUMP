# JUMP: Joint Unbiased Multimodal Preprocessing pipeline
We present a pipeline for unbiased and robust multimodal registration of 
neuroimaging modalities with minimal preprocessing.  While typical multimodal
studies need to use multiple independent processing pipelines, with diverse 
options and hyperparameters, we propose a single and structured framework to 
jointly process different image modalities. The use of state-of-the-art 
learning-based techniques enables fast inferences, which makes the presented 
method suitable for large-scale and/or multi-cohort datasets with a diverse 
number of modalities per session.

### Installation
1. Clone this repository.
2. Create a virtual environment using virtualenv or conda. We reccommend using Python>=3.8. \
```conda create -n jump-env``` \
```python -m venv jump-env```
3. Install the required dependencies under _requirements.txt_: \
```pip install -r requirements.txt```
4. You need freesurfer (v7.4+) installed.

### How to run
The 

### News and updates

15/11/2023: **JUMP is up and running** \
Initial commit of JUMP pipeline and documentation


### Citation
**A joint multimodal registration pipeline for neuroimaging with minimal preprocessing** \
Adria Casamitjana, Juan Eugenio Iglesias, Raul Tudela, Aida Niñerola-Baizan, Roser Sala-Llonch\
Submitted to ISBI'24 \
[ [article](PENDING) | [arxiv](https://arxiv.org/abs/2203.01969) | [bibtex]() ]

### Bibliography