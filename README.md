# Dependencies: If you already have an existing Torch installation and CUDA setup
1. Torchtitan:
```
cd /workspace/torchtitan
pip install -r requirements.txt
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 -U
pip3 install --pre torchdata --extra-index-url https://download.pytorch.org/whl/nightly/cu121 -U
pip install -e .

pip3 install transformers

# Remove blinker installed using apt-get to be able to pip install mlflow
apt-get --purge autoremove python3-blinker -y
pip3 install mlflow

# Accelerate and PEFT for finetuning
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
