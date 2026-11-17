# Chess Piece Classifier

A small image-classification project that predicts the color and type of one chess piece in a photo. It uses transfer learning with MobileNetV2 and has a Streamlit upload interface.

## Dataset

The dataset contains 231 JPG images in 12 classes. Labels are parsed from filenames such as `white_knight1.jpg`.

| Label | Images |
| --- | ---: |
| black_bishop | 21 |
| black_king | 16 |
| black_knight | 18 |
| black_pawn | 15 |
| black_queen | 16 |
| black_rook | 21 |
| white_bishop | 22 |
| white_king | 20 |
| white_knight | 21 |
| white_pawn | 18 |
| white_queen | 23 |
| white_rook | 20 |

The local notebook expects the extracted image folder at `chess_dataset/chess_dataset/`. In Google Colab, it can fall back to `MyDrive/Chess/chess_dataset/` if the local folder is not available.

## Run The Streamlit App

The repository's current app model and class list are `chess_model.keras` and `labels.json`.

On Windows PowerShell, from the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a JPG or PNG containing one piece, and the app will show its predicted class and model score. The score is not a guarantee of correctness.

To deploy, push `app.py`, `requirements.txt`, `chess_model.keras`, and `labels.json` to a GitHub repository, then create an app from that repository in Streamlit Community Cloud. The dataset is not needed for inference.

## Notebook Workflow

`chess.ipynb` covers filename-based labeling, stratified train/validation/test splits, image resizing with padding, a frozen MobileNetV2 baseline, exploratory data analysis, training-only augmentation, evaluation, and single-image inference.

To install the notebook dependencies in a local virtual environment, run:

```powershell
python -m pip install pandas matplotlib scikit-learn pillow tensorflow-cpu
```

The notebook saves the augmented run separately as `chess_model_augmented.keras` and `labels_augmented.json`; it does not overwrite the app model. To deploy that model instead, update `app.py` to load those filenames and upload the matching model/label pair.

## Results And Limitations

The saved notebook outputs report 37.1% test accuracy for the baseline and 34.3% for the augmented model on a 35-image test split. These are historical notebook results, not a fresh training run from this session. With only 15-23 images per class, the test estimate is noisy and the model generalizes poorly to new examples.

In this session, the dataset scan verified 231 correctly named images and the existing Keras model loaded with a 12-class output. A sample inference on `white_knight1.jpg` predicted `white_bishop` with a 30.0% score. Full notebook retraining was not verified: the VS Code notebook kernel raises a pandas C-extension initialization error, although pandas and TensorFlow import successfully from the project virtual environment in a regular Python process.

Collecting more photos with varied chess sets, viewpoints, boards, and lighting is the most useful next step for improving generalization. Keep duplicate or near-duplicate photos in the same data split to avoid an overly optimistic test score.
