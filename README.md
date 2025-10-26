# Facial Emotion Recognition

This project trains and demos facial emotion recognition models using TensorFlow/Keras and OpenCV.

Quick commands (PowerShell):

- Activate venv:
```powershell
.\facial_env\Scripts\Activate.ps1
```

- Train (short smoke run):
```powershell
.\facial_env\Scripts\python.exe .\train_model.py
```

- Run webcam demo (use model file path if needed):
```powershell
.\facial_env\Scripts\python.exe .\live_demo.py --webcam --model transfer_model_finetuned.keras
```

- Dump misclassified validation images:
```powershell
.\facial_env\Scripts\python.exe .\tools\dump_misclassified.py --model best_model.h5 --out tools/misclassified
```

- Transfer finetune example:
```powershell
.\facial_env\Scripts\python.exe .\train_transfer.py --finetune --unfreeze_layers 30 --epochs 6 --batch_size 32 --lr 1e-5
```

Files of interest:
- `train_model.py`: training script for a small CNN (grayscale 48x48)
- `train_transfer.py`: transfer learning using MobileNetV2 (RGB 96x96)
- `live_demo.py`: CLI for webcam or single-image demo
- `tools/dump_misclassified.py`: saves misclassified validation images for inspection

License: MIT (adjust as needed)
