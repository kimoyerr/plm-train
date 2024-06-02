# Dependencies
1. Torchtitan:
```
cd torchtitan
pip install -r requirements.txt
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121 -U
pip3 install --pre torchdata --index-url https://download.pytorch.org/whl/nightly -U
pip install -e .

pip install transformers
pip install mlflow
```
