import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from datasets import load_dataset
from PIL import Image

t0 = time.time()
OUT_DIR = os.environ.get("OUT_DIR", "results")
os.makedirs(OUT_DIR, exist_ok=True)
IMG_SIZE = 64
SUBSAMPLE = 6000       # scaled down from the full 23,708 for CPU-only feasibility
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
SEED = 42

device = torch.device("cpu")
print(f"Device: {device}")

print("Loading UTKFace-Cropped from local Hugging Face cache...")
ds = load_dataset("py97/UTKFace-Cropped", split="train")
print(f"Full dataset size: {len(ds)}")

rng = np.random.RandomState(SEED)
# oversample the index pool since some entries have unparseable names or a None image
pool = rng.choice(len(ds), size=min(len(ds), int(SUBSAMPLE * 1.15)), replace=False)

ages, genders, imgs = [], [], []
skipped = 0
for i in pool:
    if len(imgs) >= SUBSAMPLE:
        break
    ex = ds[int(i)]
    key = ex["__key__"]  # e.g. UTKFace/82_0_2_20170111210110290
    fname = key.split("/")[-1]
    parts = fname.split("_")
    img_obj = ex["jpg.chip.jpg"]
    if img_obj is None:
        skipped += 1
        continue
    try:
        age = int(parts[0])
        gender = int(parts[1])
    except (ValueError, IndexError):
        skipped += 1
        continue
    img = img_obj.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    imgs.append(np.asarray(img, dtype=np.float32) / 255.0)
    ages.append(age)
    genders.append(gender)

print(f"Usable samples: {len(imgs)} (skipped {skipped} with unparseable filenames)")
X = np.stack(imgs)  # (N, H, W, 3)
X = np.transpose(X, (0, 3, 1, 2))  # -> (N, 3, H, W)
y_age = np.array(ages, dtype=np.float32)
y_gender = np.array(genders, dtype=np.int64)
print(f"X shape: {X.shape}  Age range: {y_age.min()}-{y_age.max()}  Gender balance: {np.bincount(y_gender)}")

X_train, X_temp, age_train, age_temp, gen_train, gen_temp = train_test_split(
    X, y_age, y_gender, test_size=0.3, random_state=SEED, stratify=y_gender
)
X_val, X_test, age_val, age_test, gen_val, gen_test = train_test_split(
    X_temp, age_temp, gen_temp, test_size=0.5, random_state=SEED, stratify=gen_temp
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

class FaceDataset(Dataset):
    def __init__(self, X, age, gender):
        self.X = torch.tensor(X)
        self.age = torch.tensor(age)
        self.gender = torch.tensor(gender)
    def __len__(self):
        return len(self.age)
    def __getitem__(self, idx):
        return self.X[idx], self.age[idx], self.gender[idx]

train_loader = DataLoader(FaceDataset(X_train, age_train, gen_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(FaceDataset(X_val, age_val, gen_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(FaceDataset(X_test, age_test, gen_test), batch_size=BATCH_SIZE)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
        )
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(64*8*8, 128), nn.ReLU(), nn.Dropout(0.3))
        self.age_head = nn.Linear(128, 1)
        self.gender_head = nn.Linear(128, 2)
    def forward(self, x):
        h = self.shared(self.conv(x))
        return self.age_head(h).squeeze(1), self.gender_head(h)

class DNN(nn.Module):
    def __init__(self, img_size=IMG_SIZE):
        super().__init__()
        in_dim = 3 * img_size * img_size
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
        )
        self.age_head = nn.Linear(128, 1)
        self.gender_head = nn.Linear(128, 2)
    def forward(self, x):
        h = self.shared(x)
        return self.age_head(h).squeeze(1), self.gender_head(h)

def train_model(model, name):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    age_criterion = nn.L1Loss()  # MAE directly as the loss
    gender_criterion = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "val_loss": [], "val_mae": [], "val_gender_acc": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n = 0, 0
        for xb, ageb, genb in train_loader:
            xb, ageb, genb = xb.to(device), ageb.to(device), genb.to(device)
            optimizer.zero_grad()
            age_pred, gender_pred = model(xb)
            loss = age_criterion(age_pred, ageb) + gender_criterion(gender_pred, genb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(ageb)
            n += len(ageb)
        train_loss = total_loss / n

        model.eval()
        vloss, vn, all_age_pred, all_age_true, all_gen_pred, all_gen_true = 0, 0, [], [], [], []
        with torch.no_grad():
            for xb, ageb, genb in val_loader:
                xb, ageb, genb = xb.to(device), ageb.to(device), genb.to(device)
                age_pred, gender_pred = model(xb)
                loss = age_criterion(age_pred, ageb) + gender_criterion(gender_pred, genb)
                vloss += loss.item() * len(ageb)
                vn += len(ageb)
                all_age_pred.extend(age_pred.numpy()); all_age_true.extend(ageb.numpy())
                all_gen_pred.extend(gender_pred.argmax(1).numpy()); all_gen_true.extend(genb.numpy())
        val_loss = vloss / vn
        val_mae = mean_absolute_error(all_age_true, all_age_pred)
        val_gender_acc = accuracy_score(all_gen_true, all_gen_pred)
        hist["train_loss"].append(train_loss); hist["val_loss"].append(val_loss)
        hist["val_mae"].append(val_mae); hist["val_gender_acc"].append(val_gender_acc)
        print(f"[{name}] Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.3f}  val_loss={val_loss:.3f}  "
              f"val_age_MAE={val_mae:.2f}yrs  val_gender_acc={val_gender_acc:.4f}  ({time.time()-t0:.1f}s)")

    # Test evaluation
    model.eval()
    test_age_pred, test_age_true, test_gen_pred, test_gen_true = [], [], [], []
    with torch.no_grad():
        for xb, ageb, genb in test_loader:
            xb = xb.to(device)
            age_pred, gender_pred = model(xb)
            test_age_pred.extend(age_pred.numpy()); test_age_true.extend(ageb.numpy())
            test_gen_pred.extend(gender_pred.argmax(1).numpy()); test_gen_true.extend(genb.numpy())
    test_mae = mean_absolute_error(test_age_true, test_age_pred)
    test_gender_acc = accuracy_score(test_gen_true, test_gen_pred)
    print(f"[{name}] TEST  age_MAE={test_mae:.2f} years   gender_accuracy={test_gender_acc:.4f}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(hist["train_loss"], label="train"); plt.plot(hist["val_loss"], label="val")
    plt.title(f"{name} Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(hist["val_mae"]); plt.title(f"{name} Val Age MAE (years)"); plt.xlabel("Epoch")
    plt.subplot(1, 3, 3)
    plt.plot(hist["val_gender_acc"]); plt.title(f"{name} Val Gender Accuracy"); plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_training_curves.png"))
    plt.close()

    cm = confusion_matrix(test_gen_true, test_gen_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Male", "Female"], yticklabels=["Male", "Female"])
    plt.title(f"{name} Gender Confusion Matrix (test)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_gender_confusion_matrix.png"))
    plt.close()

    return test_mae, test_gender_acc

results = {}
results["CNN"] = train_model(CNN(), "CNN")
results["DNN"] = train_model(DNN(), "DNN")

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
for k, (mae, acc) in results.items():
    print(f"{k}: age MAE = {mae:.2f} years, gender accuracy = {acc:.4f}")
print(f"Subsample used: {SUBSAMPLE}/{len(ds)} images  Total elapsed: {time.time()-t0:.1f}s")

with open(os.path.join(OUT_DIR, "SUMMARY.txt"), "w") as f:
    f.write("UTKFace Age/Gender Prediction - real run results\n")
    f.write(f"Subsample used: {SUBSAMPLE} of {len(ds)} total images (CPU-only feasibility)\n")
    f.write(f"Image size: {IMG_SIZE}x{IMG_SIZE} RGB\n")
    f.write(f"Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}\n\n")
    for k, (mae, acc) in results.items():
        f.write(f"{k}: age MAE = {mae:.2f} years, gender accuracy = {acc:.4f}\n")
