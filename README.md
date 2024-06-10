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
apt-get --purge autoremove python3-blinker
pip3 install mlflow
```

# Tracking using MLFlow
Run: 
```
mlflow ui --port 5000
```
Go to http://127.0.0.1:5000 to view the dashboard
