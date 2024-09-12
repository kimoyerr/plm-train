# Dependencies: If you already have CUDA setup
1. Pre-Training using Torchtitan:
```
cd /workspace/torchtitan
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121 --upgrade
pip3 install --pre torchdata --index-url https://download.pytorch.org/whl/nightly
pip3 install -r requirements.txt
pip install -e .

pip3 install transformers

# Remove blinker installed using apt-get to be able to pip install mlflow
apt-get --purge autoremove python3-blinker -y
pip3 install mlflow

# Biopython
pip3 install biopython

# Wandb
pip3 install wandb
```
# Installation for Accelerate and PEFT for finetuning
2. Finetuning Using Accelerate and PEFT
```
pip3 install safetensors
pip3 install git+https://github.com/huggingface/accelerate
pip3 install git+https://github.com/huggingface/peft.git
pip3 install bitsandbytes
pip3 install fastcore
pip3 install biopython
pip3 install wandb
pip3 install seaborn

# Install confit
cd /workspace/ConFit
pip install -e .

pip install typing-extensions==4.12.2 

# Install DIsco
cd /workspace/DisCo-CLIP
pip install -e .
```

# Tracking using MLFlow
Run: 
```
mlflow ui --port 5000
```
Go to http://127.0.0.1:5000 to view the dashboard
