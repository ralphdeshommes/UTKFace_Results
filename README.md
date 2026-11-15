# UTKFace Age and Gender Prediction

Multi-task PyTorch models that predict a person's **age** (regression) and **gender** (classification) from one face image. A CNN and a plain fully connected network (DNN) are trained and compared.

## What it does
- Loads UTKFace (cropped) from Hugging Face (`py97/UTKFace-Cropped`) and parses age and gender from each filename.
- Uses a random **6,000-image subsample** of the 23,708 images, resized to 64x64 RGB.
- Splits 4,200 / 900 / 900 (train / validation / test), stratified by gender.
- Both models share one trunk with two heads. The loss is L1 (age) plus cross-entropy (gender). Trained 10 epochs on CPU.

## Results (900 test images)
| Model | Age error (MAE) | Gender accuracy |
|---|---|---|
| DNN | 11.49 years | 73.1% |
| CNN | 11.85 years | 71.8% |

Training curves and confusion matrices are in this repo.

## Limitations
- Only 6,000 of the 23,708 images were used, at low resolution, for 10 epochs, so these are baselines. Published models score much higher.
- The DNN edging out the CNN is within what a single run can produce; I did not repeat runs with different seeds, so no conclusion about which architecture is better.
- Age and gender labels come from filenames and the dataset's own annotations. Gender is treated as binary, as in the dataset.

## Run it
```
pip install numpy matplotlib seaborn torch scikit-learn datasets pillow
OUT_DIR=. python run_utkface.py
```
The dataset downloads automatically on first run.

## Data
UTKFace: https://susanqq.github.io/UTKFace/ (Hugging Face mirror `py97/UTKFace-Cropped`).
