# Dependencies
1. Torchtitan:
```
cd /workspace/torchtitan
pip install -r requirements.txt
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121 -U
pip3 install --pre torchdata --index-url https://download.pytorch.org/whl/nightly -U
pip install -e .

pip install transformers

# Remove blinker installed using apt-get to be able to pip install mlflow
apt-get --purge autoremove python3-blinker
pip install mlflow
```

# Tracking using MLFlow
Run: 
```
mlflow ui --port 5000
```
Go to http://127.0.0.1:5000 to view the dashboard
