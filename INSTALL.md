## Installation

### Requirements
- Linux with Python ≥ 3.9
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- Download llama-2-7b-hf model from [huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf) and place it in the root directory
- `pip install -r requirements.txt`


### Example conda environment setup
```bash
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

# under this directory
pip install -r requirements.txt

```
