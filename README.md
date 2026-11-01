# LeafGuard AI

A dark-themed Streamlit web application for binary plant-leaf health classification using a trained Ultralytics YOLO26 classification model.

## 1. Project structure

```text
LeafGuard_AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── models/
    └── best.pt             # Add your trained model here
```

## 2. Add your trained model

Copy the existing trained checkpoint into `models/` and rename it `best.pt`:

```text
models/best.pt
```

The starter project does not include model weights. Use your own trained checkpoint from:

`Plant_Disease_Project/YOLO26n_Plant_Health/weights/best.pt`

## 3. Run locally on Windows

Use Python 3.11 or 3.12 for a straightforward local setup.

Open PowerShell in the `LeafGuard_AI` project folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks environment activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

Then run `streamlit run app.py`.

The app runs locally at `http://localhost:8501` by default.

## 4. How it works

1. The app loads `models/best.pt` once and caches the model in memory.
2. The user uploads a JPG, JPEG or PNG image (maximum 10 MB).
3. The image is orientation-corrected and converted to RGB.
4. YOLO predicts on CPU using a 224-pixel input size.
5. The app displays the predicted class and probabilities for Healthy and Unhealthy.

The class indices are resolved by their names, so the application does not assume that Healthy is class 0.

## 5. Deployment notes

For Streamlit Community Cloud:

1. Push this project to a GitHub repository.
2. Include `requirements.txt`, `.streamlit/config.toml`, `app.py`, and `models/best.pt`.
3. In Streamlit Community Cloud, create an app using the repository and `app.py` as the entrypoint.
4. Select Python 3.12 in advanced settings if offered.

The model checkpoint is approximately 5.5 MB in the current training run, so it can ordinarily be committed directly to GitHub. Do not upload Kaggle credentials, API tokens, or the original dataset to the repository.

## Important limitations

- This model classifies an image as Healthy or Unhealthy. It does not identify the specific disease.
- The displayed probabilities are model outputs, not calibrated guarantees of correctness.
- Performance on photographs from outside the training dataset may differ from the held-out dataset result.
- This application is an educational screening tool, not a substitute for an agronomist or plant pathologist.
