# ============================================================
## 1. Imports
# ============================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm

# Version-safe AMP helpers (supports old/new torch.amp APIs)
try:
    from torch.amp import GradScaler as _TorchGradScaler, autocast as _torch_autocast
    _AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import GradScaler as _TorchGradScaler, autocast as _torch_autocast
    _AMP_NEW_API = False


def make_grad_scaler(enabled):
    if _AMP_NEW_API:
        return _TorchGradScaler("cuda", enabled=enabled)
    return _TorchGradScaler(enabled=enabled)


def amp_autocast(enabled):
    if _AMP_NEW_API:
        return _torch_autocast("cuda", enabled=enabled)
    return _torch_autocast(enabled=enabled)


import torchvision.transforms as T
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, matthews_corrcoef, balanced_accuracy_score
)
import gc
from scipy.linalg import cho_factor, cho_solve
from scipy import stats

torch.cuda.empty_cache()
print("All imports successful.")

# ============================================================
## 2. Device & Seed
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Seed   : {SEED}")


def cleanup_memory():
     
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



# ============================================================
## 3. Configuration
# ============================================================
CFG = {
    # Dataset paths (edit these for your environment)
    "dataset_root": "/kaggle/input/datasets/marinaeplissiti/sipakmed",
    "save_dir": "/kaggle/working/Unified_HybridVAEGAN",

    # Image parameters
    "img_size": 224,
    "batch_size": 16,

    # Model parameters
    "latent_dim": 64,

    # ---- Shared training budget (fair comparison across all models) ----
    "epochs": 40,
    "lr": 1e-4,
    "lr_critic": 1e-4,
    "weight_decay": 5e-4,

    # Unified Hybrid objective:
    #   L = lambda_rec*L_rec + lambda_kl*beta*L_KL + lambda_adv*L_adv + lambda_fm*L_FM
    # Critic trained separately with the Wasserstein loss + L_GP.
    "lambda_rec": 1.0,
    "lambda_kl": 1.0,
    "lambda_adv": 0.05,
    "lambda_fm": 0.5,
    "lambda_gp": 10.0,

    # Beta-VAE schedule
    "beta_start": 0.0,
    "beta_end": 3.0,
    "beta_warmup": 12,
    "beta_tail_lock": 8,
    "beta_cyclical": False,          # monotonic warm-up; no mid-training resets

    # Keeps the adversarial/feature-matching gradient from reshaping the
    # Encoder's latent space (the GMM/Mahalanobis anomaly score is
    # computed purely from that latent space). Evidence from prior runs
    # favors False; kept at its original default (not re-chosen against
    # test data).
    "stop_grad_encoder_from_gan": False,

    "n_critic": 5,                  # critic steps per generator/encoder step
    "patience": 10,                 # Standalone-VAE, Standalone-GAN, CBiGAN
    "patience_hybrid": 20,          # Hybrid adversarial training is noisier
    "adv_warmup_epochs": 8,         # <<< single controlled change (was 6):
                                     #     pure-VAE warm-up before the GAN
                                     #     term is switched on.

    # CBiGAN (Carrara et al., arXiv:2011.08102) hyperparameters
    "cbigan_alpha": 0.10,           # consistency-loss weight, Eq. 13
    "cbigan_lambda": 0.10,          # anomaly-score mix weight, Eq. 7

    # Classes
    "normal_classes": [
        "im_Parabasal",
        "im_Superficial-Intermediate",
    ],
    "abnormal_classes": [
        "im_Dyskeratotic",
        "im_Koilocytotic",
        "im_Metaplastic",
    ],
}

os.makedirs(CFG["save_dir"], exist_ok=True)
print(f"Save directory : {CFG['save_dir']}")
print(f"Single controlled change vs prior successful run: "
      f"adv_warmup_epochs = {CFG['adv_warmup_epochs']} (was 6)")


# ============================================================
## 4. Locate Dataset
# ============================================================
if not os.path.exists(CFG["dataset_root"]):
    raise FileNotFoundError(f"Dataset not found at: {CFG['dataset_root']}")
else:
    print(f"Dataset found at: {CFG['dataset_root']}")
    print(f"  Contents: {os.listdir(CFG['dataset_root'])[:10]}")



# ============================================================
## 5. Image Loader (path-first, group-aware split)
# ============================================================
def list_files_with_groups(class_list, root):
    paths, labels, groups = [], [], []
    for cls in class_list:
        cropped = os.path.join(root, cls, cls, "CROPPED")
        cropped_alt = os.path.join(root, cls, "CROPPED")
        flat = os.path.join(root, cls)

        if os.path.exists(cropped):
            path = cropped
        elif os.path.exists(cropped_alt):
            path = cropped_alt
        else:
            path = flat
            if not os.path.exists(path):
                print(f"  Not found: {cls}")
                continue

        files = glob(os.path.join(path, "*.bmp")) + \
            glob(os.path.join(path, "*.png")) + \
            glob(os.path.join(path, "*.jpg"))

        if len(files) == 0:
            files = glob(os.path.join(path, "*", "*.bmp")) + \
                glob(os.path.join(path, "*", "*.png")) + \
                glob(os.path.join(path, "*", "*.jpg"))

        print(f"  {cls}: {len(files)} files")

        for f in files:
            paths.append(f)
            labels.append(cls)
            groups.append(os.path.splitext(os.path.basename(f))[0].split('_')[0])

    return np.array(paths), np.array(labels), np.array(groups)


def load_images_from_paths(paths, img_size=224):
    images = []
    for f in tqdm(paths, desc="Loading", leave=False):
        try:
            if f.endswith('.dat'):
                raw = np.loadtxt(f)
                if raw.ndim == 1:
                    s = int(np.sqrt(len(raw)))
                    raw = raw.reshape(s, s)
                raw = np.clip(raw, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.imread(f)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.GaussianBlur(img, (3, 3), 1.0)
            img = img.astype(np.float32) / 255.0
            images.append(img)
        except Exception:
            continue

    return np.array(images, dtype=np.float32)


# ============================================================
## 6. Group-Aware Data Split (no parent-image leakage)
# ============================================================
print("\nListing files...")
normal_paths, normal_labels, normal_groups = list_files_with_groups(
    CFG["normal_classes"], CFG["dataset_root"]
)
abnormal_paths, abnormal_labels, abnormal_groups = list_files_with_groups(
    CFG["abnormal_classes"], CFG["dataset_root"]
)

print(f"\nNormal files   : {len(normal_paths)}  ({len(set(normal_groups))} unique)")
print(f"Abnormal files : {len(abnormal_paths)}  ({len(set(abnormal_groups))} unique)")


def group_split(paths, labels, groups, test_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_a, idx_b = next(gss.split(paths, labels, groups=groups))
    return paths[idx_a], paths[idx_b], labels[idx_a], labels[idx_b], groups[idx_a], groups[idx_b]


# Split normal (this is the ONLY data used for training -- healthy-only)
p_tr_n, p_tmp_n, l_tr_n, l_tmp_n, g_tr_n, g_tmp_n = group_split(
    normal_paths, normal_labels, normal_groups, test_size=0.30, seed=SEED
)
p_val_n, p_tst_n, l_val_n, l_tst_n, g_val_n, g_tst_n = group_split(
    p_tmp_n, l_tmp_n, g_tmp_n, test_size=0.50, seed=SEED
)

# Split abnormal (used ONLY for validation/test of the anomaly detector)
p_tr_a, p_tmp_a, l_tr_a, l_tmp_a, g_tr_a, g_tmp_a = group_split(
    abnormal_paths, abnormal_labels, abnormal_groups, test_size=0.40, seed=SEED
)
p_val_a, p_tst_a, l_val_a, l_tst_a, g_val_a, g_tst_a = group_split(
    p_tmp_a, l_tmp_a, g_tmp_a, test_size=0.50, seed=SEED
)

# Sanity check: no group overlap between splits
assert len(set(g_tr_n) & set(g_val_n)) == 0
assert len(set(g_tr_n) & set(g_tst_n)) == 0
assert len(set(g_val_n) & set(g_tst_n)) == 0
assert len(set(g_tr_a) & set(g_val_a)) == 0
assert len(set(g_tr_a) & set(g_tst_a)) == 0
assert len(set(g_val_a) & set(g_tst_a)) == 0
print("No parent-image leakage between splits.")

print("\nLoading images...")
X_tr_n = load_images_from_paths(p_tr_n, CFG["img_size"])
X_val_n = load_images_from_paths(p_val_n, CFG["img_size"])
X_tst_n = load_images_from_paths(p_tst_n, CFG["img_size"])

X_tr_a = load_images_from_paths(p_tr_a, CFG["img_size"])
X_val_a = load_images_from_paths(p_val_a, CFG["img_size"])
X_tst_a = load_images_from_paths(p_tst_a, CFG["img_size"])

X_val = np.concatenate([X_val_n, X_val_a])
y_val = np.concatenate([np.zeros(len(X_val_n)), np.ones(len(X_val_a))])
X_test = np.concatenate([X_tst_n, X_tst_a])
y_test = np.concatenate([np.zeros(len(X_tst_n)), np.ones(len(X_tst_a))])

normal_imgs = np.concatenate([X_tr_n, X_val_n, X_tst_n])
normal_labels_all = np.concatenate([l_tr_n, l_val_n, l_tst_n])
abnormal_imgs = np.concatenate([X_tr_a, X_val_a, X_tst_a])
abnormal_labels_all = np.concatenate([l_tr_a, l_val_a, l_tst_a])

print("\nSplit Summary:")
print(f"  Train (healthy only) : {len(X_tr_n)}")
print(f"  Val   (n + a)        : {len(X_val)}")
print(f"  Test  (n + a)        : {len(X_test)}")

del p_tmp_n, p_tmp_a, p_tr_n, p_tr_a, p_val_n, p_val_a, p_tst_n, p_tst_a
del normal_paths, abnormal_paths, normal_groups, abnormal_groups
del g_tr_n, g_tmp_n, g_val_n, g_tst_n, g_tr_a, g_tmp_a, g_val_a, g_tst_a
gc.collect()
print("Cleaned temporary arrays.")

# ============================================================
## 7. Utility Classes
# ============================================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None

    def step(self, loss, model):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print(f"  Best model restored (loss={self.best_loss:.5f})")


# ============================================================
## 8. Augmentation Pipeline
# ============================================================
def build_light_augment_transforms():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.3),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=8),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    ])


# ============================================================
## 9. Dataset Class
# ============================================================
class CellDataset(Dataset):
    def __init__(self, images, labels=None, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.transforms = build_light_augment_transforms() if augment else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float()
        if self.augment and self.transforms is not None:
            x = self.transforms(x)
        if self.labels is not None:
            return x, torch.tensor(self.labels[idx]).float()
        return x



# ============================================================
## 10. DataLoaders
# ============================================================
bs = CFG["batch_size"]

loader_train = DataLoader(
    CellDataset(X_tr_n, augment=True),
    batch_size=bs, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()
)
loader_val_n = DataLoader(
    CellDataset(X_val_n),
    batch_size=bs, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
)
loader_val_a = DataLoader(
    CellDataset(X_val_a),
    batch_size=bs, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
)
loader_test = DataLoader(
    CellDataset(X_test, y_test),
    batch_size=bs, shuffle=False
)

print("All DataLoaders ready.")


# %%

# ============================================================
## 11. Initial Visualization
# ============================================================
def plot_samples(normal, abnormal, n=8, save=None):
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 5))
    for i in range(n):
        axes[0, i].imshow(normal[random.randint(0, len(normal) - 1)])
        axes[0, i].axis("off")
        axes[1, i].imshow(abnormal[random.randint(0, len(abnormal) - 1)])
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Normal", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Abnormal", fontsize=12, fontweight='bold')
    plt.suptitle("SIPaKMeD - Dataset Samples", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()


def plot_distribution(n_labels, a_labels, save=None):
    counts = Counter(np.concatenate([n_labels, a_labels]))
    classes = list(counts.keys())
    values = list(counts.values())
    colors = ['#2196F3' if c in CFG["normal_classes"] else '#F44336' for c in classes]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(classes, values, color=colors, edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                  str(val), ha='center', fontsize=10)
    plt.xticks(rotation=20, ha='right')
    plt.ylabel("Count")
    plt.title("Class Distribution", fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()


plot_samples(normal_imgs, abnormal_imgs,
             save=os.path.join(CFG["save_dir"], "01_dataset_samples.png"))
plot_distribution(normal_labels_all, abnormal_labels_all,
                   save=os.path.join(CFG["save_dir"], "02_class_distribution.png"))

del normal_imgs, abnormal_imgs, normal_labels_all, abnormal_labels_all
gc.collect()
print("Cleaned visualization arrays.")


# ============================================================
## 12. Shared Building Blocks
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.attn = ChannelAttention(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.act(x + self.dropout(self.attn(self.block(x))))


class SelfAttention2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        attn = torch.softmax(torch.bmm(q, k) / (C ** 0.5), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out




# ============================================================
## 13. Models
# ============================================================
class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU())
        self.res1 = ResidualBlock(64)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.GELU())
        self.res2 = ResidualBlock(128)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.GELU())
        self.res3 = ResidualBlock(256)
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.GELU())
        self.res4 = ResidualBlock(512)

        self.flat_dim = 512 * 14 * 14
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.res1(self.enc1(x))
        h = self.res2(self.enc2(h))
        h = self.res3(self.enc3(h))
        h = self.res4(self.enc4(h))
        h = h.reshape(h.size(0), -1)
        mu = self.fc_mu(h).clamp(-50, 50)
        logvar = self.fc_logvar(h).clamp(-10, 10)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        mu = mu.clamp(-50, 50)
        logvar = logvar.clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std


class VAEDecoder(nn.Module):
    
    def __init__(self, latent_dim=64):
        super().__init__()
        self.flat_dim = 512 * 14 * 14
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.GELU())
        self.res_d4 = ResidualBlock(256)
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.GELU())
        self.res_d3 = ResidualBlock(128)
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU())
        self.res_d2 = ResidualBlock(64)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Sigmoid())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        h = self.fc_dec(z).view(-1, 512, 14, 14)
        h = self.res_d4(self.dec4(h))
        h = self.res_d3(self.dec3(h))
        h = self.res_d2(self.dec2(h))
        return self.dec1(h)


class StandaloneVAE(nn.Module):
    
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


class ImprovedGenerator(nn.Module):
    
    def __init__(self, latent_dim=64):
        super().__init__()
        self.flat_dim = 512 * 14 * 14
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.GELU())
        self.res_d4 = ResidualBlock(256)
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.GELU())
        self.res_d3 = ResidualBlock(128)
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU())
        self.res_d2 = ResidualBlock(64)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Sigmoid())

    def forward(self, z):
        h = self.fc_dec(z).view(-1, 512, 14, 14)
        h = self.res_d4(self.dec4(h))
        h = self.res_d3(self.dec3(h))
        h = self.res_d2(self.dec2(h))
        return self.dec1(h)


class ImprovedCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            SelfAttention2d(128),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            spectral_norm(nn.Conv2d(512, 512, 4, 2, 1)),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(512, 1))
        )

    def forward(self, x):
        return self.net(x)


def gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interp, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()




# ============================================================
## 14. Loss Functions (shared)
# ============================================================
def reconstruction_loss(recon, x):
    mse = F.mse_loss(recon, x, reduction='mean')
    mae = F.l1_loss(recon, x, reduction='mean')
    return 0.6 * mse + 0.4 * mae


def kl_divergence(mu, logvar, free_bits=0.5):
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.mean()


def cyclical_beta(epoch, cfg, beta_max=1.0):
    """Linear warmup to beta_max, then (if beta_cyclical=False) holds at
    beta_max for the rest of training with no resets. The last
    beta_tail_lock epochs are always pinned to beta_max."""
    beta_start = cfg.get("beta_start", 0.0)
    warmup = max(0, int(cfg.get("beta_warmup", 0)))
    epochs = cfg["epochs"]
    tail_lock = max(0, int(cfg.get("beta_tail_lock", 2)))
    cyclical_enabled = cfg.get("beta_cyclical", True)

    if warmup > 0 and epoch < warmup:
        ratio = epoch / max(warmup - 1, 1)
        return beta_start + ratio * (beta_max - beta_start)

    if not cyclical_enabled:
        return beta_max

    if epoch >= epochs - tail_lock:
        return beta_max

    epoch_after_warmup = epoch - warmup
    remaining_epochs = max(epochs - warmup, 1)

    if remaining_epochs < 4:
        return beta_max

    cycle_length = max(4, remaining_epochs // 3)
    cycle = epoch_after_warmup % cycle_length
    ratio = cycle / (cycle_length / 2)
    return beta_start + min(ratio, 1.0) * (beta_max - beta_start)


def validate_beta_schedule(cfg, beta_max=None, verbose=True):
    """Cheap, GPU-free sanity check for the beta schedule before training."""
    bmax = beta_max if beta_max is not None else cfg.get("beta_end", 1.0)
    betas = [cyclical_beta(e, cfg, bmax) for e in range(cfg["epochs"])]
    if verbose:
        mode = "cyclical" if cfg.get("beta_cyclical", True) else "monotonic (no cycling after warmup)"
        print(f"Beta schedule check (epochs={cfg['epochs']}, "
              f"warmup={cfg.get('beta_warmup', 0)}, "
              f"tail_lock={cfg.get('beta_tail_lock', 2)}, mode={mode}, beta_max={bmax}):")
        for e, b in enumerate(betas):
            marker = "  <-- last epoch" if e == cfg["epochs"] - 1 else ""
            print(f"  epoch {e:3d}: beta={b:.3f}{marker}")
        if betas[-1] < bmax * 0.95:
            print("  WARNING: last epoch does not sit at beta_max -- "
                  "check beta_warmup / beta_tail_lock before training.")
        else:
            print("  OK: last epoch is at beta_max. Safe to train.")
    return betas


def feature_matching_loss(critic, real, fake):
   
    feats_real = real
    feats_fake = fake
    loss = 0.0
    n_layers = 0
    for layer in critic.net[:-2]:  # skip Flatten + final Linear
        feats_real = layer(feats_real)
        feats_fake = layer(feats_fake)
        loss = loss + F.l1_loss(feats_fake, feats_real.detach())
        n_layers += 1
    return loss / max(n_layers, 1)


# ============================================================
## 14b. Shared per-epoch checkpoint-selection helper
#
#     Cheap, task-aligned validation signal used to pick which training
#     epoch to keep, for Hybrid AND Standalone-VAE (Standalone-WGAN-GP
#     and CBiGAN already used a val-AUC criterion; this makes all four
#     models consistent). NOT the final reported GMM -- that is the full
#     k/covariance sweep in Section 22, run once after training on the
#     restored best checkpoint.
# ============================================================
def _encode_mu_all(encoder, loader):
    encoder.eval()
    mus = []
    with torch.no_grad():
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            mu, _ = encoder(x)
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


def _quick_gmm_val_auc(encoder, X_tr_n, loader_val_n, loader_val_a, k=2, batch_size=32):
    was_training = encoder.training
    encoder.eval()

    train_loader = DataLoader(CellDataset(X_tr_n), batch_size=batch_size, shuffle=False)
    z_train = _encode_mu_all(encoder, train_loader)
    z_vn = _encode_mu_all(encoder, loader_val_n)
    z_va = _encode_mu_all(encoder, loader_val_a)

    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED,
                           reg_covar=1e-4, n_init=1, max_iter=100).fit(z_train)
    chols = [cho_factor(cov, lower=True) for cov in gmm.covariances_]

    def _mahal(z):
        dists = []
        for mean, chol in zip(gmm.means_, chols):
            diff = z - mean
            sol = cho_solve(chol, diff.T)
            d2 = np.einsum('ij,ji->i', diff, sol)
            dists.append(np.sqrt(np.maximum(d2, 0.0)))
        return np.min(np.stack(dists, axis=0), axis=0)

    s_vn, s_va = _mahal(z_vn), _mahal(z_va)
    y_val = np.concatenate([np.zeros(len(s_vn)), np.ones(len(s_va))])
    auc = roc_auc_score(y_val, np.concatenate([s_vn, s_va]))

    if was_training:
        encoder.train()
    return auc




# ============================================================
## 15. Proposed Model -- Unified End-to-End Training (Hybrid beta-VAE + WGAN-GP)
# ============================================================
def train_hybrid_vae_wgan_gp(loader_train, loader_val_n, loader_val_a, X_tr_n, cfg):
     
    encoder = VAEEncoder(cfg["latent_dim"]).to(device)
    decoder = ImprovedGenerator(cfg["latent_dim"]).to(device)
    critic = ImprovedCritic().to(device)

    opt_EDG = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    opt_D = optim.Adam(critic.parameters(), lr=cfg["lr_critic"], betas=(0.0, 0.9))
    scheduler_EDG = optim.lr_scheduler.CosineAnnealingLR(opt_EDG, T_max=cfg["epochs"], eta_min=1e-6)

    use_amp = torch.cuda.is_available()
    scaler_EDG = make_grad_scaler(use_amp)
    scaler_D = make_grad_scaler(use_amp)

    adv_warmup = max(0, int(cfg.get("adv_warmup_epochs", 0)))
    lambda_fm = cfg.get("lambda_fm", 0.0)
    stop_grad_enc = cfg.get("stop_grad_encoder_from_gan", False)
    patience_hybrid = cfg.get("patience_hybrid", cfg["patience"])

    history = {"rec": [], "kl": [], "adv": [], "fm": [], "critic": [], "gp": [],
               "val_loss": [], "val_auc": [], "beta": []}

    best_val_auc = -1.0
    best_state = None
    no_improve = 0

    print("\n" + "=" * 70)
    print("PHASE 1 -- UNIFIED END-TO-END TRAINING (Hybrid beta-VAE + WGAN-GP)")
    print(f"lambda_adv={cfg['lambda_adv']} | beta_end={cfg['beta_end']} | "
          f"adv_warmup_epochs={adv_warmup} | patience_hybrid={patience_hybrid} | "
          f"lambda_fm={lambda_fm} | stop_grad_encoder_from_gan={stop_grad_enc}")
    print("Checkpoint selection: validation AUC (quick GMM, k=2/full), warmup included")
    print("=" * 70)

    for epoch in range(cfg["epochs"]):
        beta = cyclical_beta(epoch, cfg, beta_max=cfg["beta_end"])
        adv_active = epoch >= adv_warmup
        encoder.train(); decoder.train(); critic.train()

        ep_rec = ep_kl = ep_adv = ep_fm = ep_c = ep_gp = 0.0
        n_batches = n_critic_steps = 0
        desc = f"Unified {epoch + 1}/{cfg['epochs']}" + (" [VAE warmup]" if not adv_active else "")

        for x in tqdm(loader_train, desc=desc, leave=False):
            x = (x[0] if isinstance(x, (list, tuple)) else x).to(device)

            if adv_active:
                for _ in range(cfg["n_critic"]):
                    with torch.no_grad():
                        mu, logvar = encoder(x)
                        x_recon = decoder(encoder.reparameterize(mu, logvar))
                    with amp_autocast(use_amp):
                        d_real = critic(x).mean()
                        d_fake = critic(x_recon).mean()
                    gp = gradient_penalty(critic, x.float(), x_recon.float(), device)
                    loss_D = d_fake - d_real + cfg["lambda_gp"] * gp

                    opt_D.zero_grad(set_to_none=True)
                    scaler_D.scale(loss_D).backward()
                    scaler_D.unscale_(opt_D)
                    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    scaler_D.step(opt_D)
                    scaler_D.update()

                    ep_c += (d_real - d_fake).item()
                    ep_gp += gp.item()
                    n_critic_steps += 1

            if adv_active:
                for p in critic.parameters():
                    p.requires_grad_(False)

            with amp_autocast(use_amp):
                mu, logvar = encoder(x)
                z = encoder.reparameterize(mu, logvar)
                x_recon = decoder(z)
                rec_loss = reconstruction_loss(x_recon, x)
                kl_loss = kl_divergence(mu, logvar)

                if adv_active:
                    z_for_gan = z.detach() if stop_grad_enc else z
                    x_recon_gan = decoder(z_for_gan) if stop_grad_enc else x_recon
                    adv_loss = -critic(x_recon_gan).mean()
                    fm_loss = feature_matching_loss(critic, x, x_recon_gan)
                    total_loss = (cfg["lambda_rec"] * rec_loss + cfg["lambda_kl"] * beta * kl_loss
                                  + cfg["lambda_adv"] * adv_loss + lambda_fm * fm_loss)
                else:
                    adv_loss = torch.zeros((), device=device)
                    fm_loss = torch.zeros((), device=device)
                    total_loss = cfg["lambda_rec"] * rec_loss + cfg["lambda_kl"] * beta * kl_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("  NaN/Inf detected! Skipping batch.")
                if adv_active:
                    for p in critic.parameters():
                        p.requires_grad_(True)
                continue

            opt_EDG.zero_grad(set_to_none=True)
            scaler_EDG.scale(total_loss).backward()
            scaler_EDG.unscale_(opt_EDG)
            nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            scaler_EDG.step(opt_EDG)
            scaler_EDG.update()

            if adv_active:
                for p in critic.parameters():
                    p.requires_grad_(True)

            ep_rec += rec_loss.item(); ep_kl += kl_loss.item()
            ep_adv += adv_loss.item(); ep_fm += fm_loss.item()
            n_batches += 1

        scheduler_EDG.step()

        history["rec"].append(ep_rec / max(n_batches, 1))
        history["kl"].append(ep_kl / max(n_batches, 1))
        history["adv"].append(ep_adv / max(n_batches, 1))
        history["fm"].append(ep_fm / max(n_batches, 1))
        history["critic"].append(ep_c / max(n_critic_steps, 1))
        history["gp"].append(ep_gp / max(n_critic_steps, 1))
        history["beta"].append(beta)

        # ---- Validation: healthy-only reconstruction+KL loss (reporting only) ----
        encoder.eval(); decoder.eval()
        val_loss_total, n_val_batches = 0.0, 0
        with torch.no_grad():
            for xv in loader_val_n:
                xv = (xv[0] if isinstance(xv, (list, tuple)) else xv).to(device)
                mu, logvar = encoder(xv)
                x_recon = decoder(mu)
                rec_loss_v = reconstruction_loss(x_recon, xv)
                kl_loss_v = kl_divergence(mu, logvar)
                val_loss_total += (cfg["lambda_rec"] * rec_loss_v + cfg["lambda_kl"] * cfg["beta_end"] * kl_loss_v).item()
                n_val_batches += 1
        val_loss_total /= max(n_val_batches, 1)
        history["val_loss"].append(val_loss_total)

        # ---- Checkpoint selection: validation AUC (task-aligned), warmup included ----
        val_auc = _quick_gmm_val_auc(encoder, X_tr_n, loader_val_n, loader_val_a, k=2)
        history["val_auc"].append(val_auc)

        tag = " [VAE warmup]" if not adv_active else ""
        print(f"Epoch {epoch + 1:03d}{tag} | Beta={beta:.3f} | Rec={history['rec'][-1]:.4f} | "
              f"KL={history['kl'][-1]:.4f} | Adv={history['adv'][-1]:.4f} | FM={history['fm'][-1]:.4f} | "
              f"W_dist={history['critic'][-1]:.4f} | Val_Loss={val_loss_total:.4f} | Val_AUC={val_auc:.4f}")

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()},
                "critic": {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()},
            }
            no_improve = 0
            print(f"  Best hybrid checkpoint saved (Val_AUC={val_auc:.4f})")
        else:
            no_improve += 1
            # Never early-stop during the mandatory warmup, matching the
            # original protocol -- only the comparison itself now also
            # includes warmup epochs (previously it did not).
            if no_improve >= patience_hybrid and adv_active:
                print(f"  Early stopping at epoch {epoch + 1} "
                      f"(no Val_AUC improvement for {patience_hybrid} epochs)")
                break

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])
        critic.load_state_dict(best_state["critic"])
        print(f"Best hybrid model restored (Val_AUC={best_val_auc:.4f})")

    torch.save(encoder.state_dict(), os.path.join(cfg["save_dir"], "hybrid_encoder.pth"))
    torch.save(decoder.state_dict(), os.path.join(cfg["save_dir"], "hybrid_decoder.pth"))
    torch.save(critic.state_dict(), os.path.join(cfg["save_dir"], "hybrid_critic.pth"))

    encoder.eval(); decoder.eval(); critic.eval()
    print(f"Hybrid end-to-end training complete. Best validation AUC={best_val_auc:.5f}")
    return encoder, decoder, critic, history




# ============================================================
## 16. Baseline #1 -- Standalone beta-VAE (no adversarial term)
# ============================================================
def train_standalone_vae(loader_train, loader_val_n, loader_val_a, X_tr_n, cfg):
     
    model = StandaloneVAE(cfg["latent_dim"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    use_amp = torch.cuda.is_available()
    scaler = make_grad_scaler(use_amp)
    history = {"train": [], "val_loss": [], "val_auc": [], "beta": []}

    best_val_auc = -1.0
    best_state = None
    no_improve = 0

    print("\n" + "=" * 70)
    print("BASELINE 1 -- STANDALONE beta-VAE (no adversarial component)")
    print("Checkpoint selection: validation AUC (quick GMM, k=2/full)")
    print("=" * 70)

    for epoch in range(cfg["epochs"]):
        beta = cyclical_beta(epoch, cfg, beta_max=cfg["beta_end"])
        model.train()
        tr_loss = 0.0
        for x in tqdm(loader_train, desc=f"VAE-only {epoch+1}/{cfg['epochs']}", leave=False):
            x = (x[0] if isinstance(x, (list, tuple)) else x).to(device)
            optimizer.zero_grad()
            with amp_autocast(use_amp):
                recon, mu, logvar = model(x)
                rec_loss = reconstruction_loss(recon, x)
                kl_loss = kl_divergence(mu, logvar)
                loss = rec_loss + beta * kl_loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()
        tr_loss /= len(loader_train)
        history["train"].append(tr_loss)
        history["beta"].append(beta)

        # ---- Healthy validation reconstruction+KL loss (reporting only) ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xv in loader_val_n:
                xv = (xv[0] if isinstance(xv, (list, tuple)) else xv).to(device)
                recon, mu, logvar = model(xv)
                rec_loss = reconstruction_loss(recon, xv)
                kl_loss = kl_divergence(mu, logvar)
                val_loss += (rec_loss + cfg["beta_end"] * kl_loss).item()
        val_loss /= len(loader_val_n)
        history["val_loss"].append(val_loss)
        scheduler.step()

        # ---- Checkpoint selection: validation AUC (task-aligned) ----
        val_auc = _quick_gmm_val_auc(model.encoder, X_tr_n, loader_val_n, loader_val_a, k=2)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch+1:03d} | Beta={beta:.3f} | Train={tr_loss:.5f} | "
              f"Val_Loss={val_loss:.5f} | Val_AUC={val_auc:.4f}")

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "standalone_vae.pth"))
            print(f"  Best standalone-VAE checkpoint saved (Val_AUC={val_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(no Val_AUC improvement for {cfg['patience']} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best standalone-VAE model restored (Val_AUC={best_val_auc:.4f})")

    model.eval()
    print(f"Standalone beta-VAE training complete. Best Val_AUC={best_val_auc:.5f}")
    return model, history




# ============================================================
## 17. Baseline #2 -- Standalone WGAN-GP (prior latent only, no VAE encoder)
# ============================================================
def train_standalone_gan(loader_train, loader_val_n, loader_val_a, cfg):
    generator = ImprovedGenerator(cfg["latent_dim"]).to(device)
    critic = ImprovedCritic().to(device)
    opt_G = optim.Adam(generator.parameters(), lr=cfg["lr_critic"], betas=(0.0, 0.9))
    opt_C = optim.Adam(critic.parameters(), lr=cfg["lr_critic"], betas=(0.0, 0.9))
    use_amp = torch.cuda.is_available()
    scaler_G = make_grad_scaler(use_amp)
    scaler_C = make_grad_scaler(use_amp)
    history = {"W_dist": [], "G_loss": [], "GP": [], "val_auc": []}

    best_val_auc = -1.0
    best_state = None
    no_improve = 0

    print("\n" + "=" * 70)
    print("BASELINE 2 -- STANDALONE WGAN-GP (no VAE encoder), val-AUC checkpointing")
    print("=" * 70)

    @torch.no_grad()
    def _score(loader):
        critic.eval()
        out = []
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            d = critic(x).squeeze(-1)
            out.append((-d).cpu().numpy())
        critic.train()
        return np.concatenate(out)

    for epoch in range(cfg["epochs"]):
        generator.train(); critic.train()
        ep = {"W_dist": 0.0, "G_loss": 0.0, "GP": 0.0}
        nb_ = 0; ncs = 0
        for real in tqdm(loader_train, desc=f"GAN-only {epoch+1}/{cfg['epochs']}", leave=False):
            real = (real[0] if isinstance(real, (list, tuple)) else real).to(device)
            b = real.size(0)
            for _ in range(cfg["n_critic"]):
                z = torch.randn(b, cfg["latent_dim"], device=device)
                with amp_autocast(use_amp):
                    fake = generator(z).detach()
                    loss_real = -critic(real).mean()
                    loss_fake = critic(fake).mean()
                gp = gradient_penalty(critic, real.float(), fake.float(), device)
                loss_C = loss_real + loss_fake + cfg["lambda_gp"] * gp
                opt_C.zero_grad()
                scaler_C.scale(loss_C).backward()
                scaler_C.unscale_(opt_C)
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                scaler_C.step(opt_C)
                scaler_C.update()
                ep["W_dist"] += -(loss_real + loss_fake).item()
                ep["GP"] += gp.item()
                ncs += 1

            z = torch.randn(b, cfg["latent_dim"], device=device)
            with amp_autocast(use_amp):
                fake = generator(z)
                loss_G = -critic(fake).mean()
            opt_G.zero_grad()
            scaler_G.scale(loss_G).backward()
            scaler_G.unscale_(opt_G)
            nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            scaler_G.step(opt_G)
            scaler_G.update()
            ep["G_loss"] += loss_G.item()
            nb_ += 1

        history["W_dist"].append(ep["W_dist"] / max(ncs, 1))
        history["GP"].append(ep["GP"] / max(ncs, 1))
        history["G_loss"].append(ep["G_loss"] / max(nb_, 1))

        s_vn, s_va = _score(loader_val_n), _score(loader_val_a)
        y_val = np.concatenate([np.zeros(len(s_vn)), np.ones(len(s_va))])
        s_val = np.concatenate([s_vn, s_va])
        val_auc = roc_auc_score(y_val, s_val)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch+1:03d} | W_dist={history['W_dist'][-1]:.4f} | "
              f"G_loss={history['G_loss'][-1]:.4f} | GP={history['GP'][-1]:.4f} | "
              f"Val_AUC={val_auc:.4f}")

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {
                "G": {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()},
                "C": {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()},
            }
            no_improve = 0
            print(f"  Best standalone-GAN checkpoint saved (Val_AUC={val_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1} (no Val_AUC improvement for {cfg['patience']} epochs)")
                break

    if best_state is not None:
        generator.load_state_dict(best_state["G"])
        critic.load_state_dict(best_state["C"])
        print(f"Best model restored (Val_AUC={best_val_auc:.4f})")

    torch.save(generator.state_dict(), os.path.join(cfg["save_dir"], "standalone_gan_generator.pth"))
    torch.save(critic.state_dict(), os.path.join(cfg["save_dir"], "standalone_gan_critic.pth"))
    print("Standalone WGAN-GP training complete.")
    return generator, critic, history



# ============================================================
## 18. Run -- Train Hybrid, Standalone-VAE, Standalone-GAN
# ============================================================
print("\n" + "=" * 70)
print("PRE-FLIGHT CHECK -- beta schedule (no GPU cost)")
print("=" * 70)
validate_beta_schedule(CFG, beta_max=CFG["beta_end"])

hybrid_encoder, hybrid_generator, hybrid_critic, hybrid_history = train_hybrid_vae_wgan_gp(
    loader_train, loader_val_n, loader_val_a, X_tr_n, CFG
)
cleanup_memory()
standalone_vae, vae_history = train_standalone_vae(
    loader_train, loader_val_n, loader_val_a, X_tr_n, CFG
)
cleanup_memory()
standalone_gan_G, standalone_gan_C, gan_history = train_standalone_gan(
    loader_train, loader_val_n, loader_val_a, CFG
)
cleanup_memory()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(hybrid_history["rec"], label="Rec", lw=2)
axes[0].plot(hybrid_history["kl"], label="KL", lw=2)
axes[0].plot(hybrid_history["adv"], label="Adv", lw=2)
axes[0].plot(hybrid_history["fm"], label="FeatureMatch", lw=2)
axes[0].set_title("Hybrid -- Loss Components", fontweight="bold")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(hybrid_history["critic"], color="#6a1b9a", lw=2)
axes[1].set_title("Hybrid -- Wasserstein Distance", fontweight="bold")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CFG["save_dir"], "03_training_curves.png"), dpi=300, bbox_inches='tight')
plt.show()



# ============================================================
## 19. Healthy Latent Codes (for GMM fitting below)
# ============================================================
def get_latent_codes(encode_fn, data, batch_size=32):
    loader = DataLoader(CellDataset(data), batch_size=batch_size, shuffle=False)
    mus = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            mu = encode_fn(x)
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


hybrid_encoder.eval()


def hybrid_encode_mu(x):
    mu, _ = hybrid_encoder(x)
    return mu


standalone_vae.eval()


def standalone_vae_encode_mu(x):
    mu, _ = standalone_vae.encoder(x)
    return mu


z_healthy_hybrid = get_latent_codes(hybrid_encode_mu, X_tr_n)
z_healthy_vae = get_latent_codes(standalone_vae_encode_mu, X_tr_n)
print(f"Healthy latent codes extracted: Hybrid {z_healthy_hybrid.shape}, "
      f"Standalone-VAE {z_healthy_vae.shape}")



# ============================================================
## 20. Anomaly Score Computation (GAN baseline only -- Hybrid/VAE use the
##     GMM-Mahalanobis machinery defined in Section 21 below)
# ============================================================
@torch.no_grad()
def compute_standalone_gan_scores(critic, loader):
     
    critic.eval()
    scores = []
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        d = critic(x).squeeze(-1)
        scores.append((-d).cpu().numpy())
    return np.concatenate(scores)




# ============================================================
## 21. Evaluation Helpers
# ============================================================
def find_optimal_threshold(scores_n, scores_a):
    y_true = np.concatenate([np.zeros(len(scores_n)), np.ones(len(scores_a))])
    y_score = np.concatenate([scores_n, scores_a])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return float(thresholds[np.argmax(tpr - fpr)])


def bootstrap_ci(y_true, y_score, metric_fn, n_bootstrap=1000, ci=0.95):
    rng = np.random.default_rng(SEED)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except ValueError:
            continue
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    alpha = (1 - ci) / 2
    return float(np.mean(scores)), float(np.percentile(scores, 100 * alpha)), float(np.percentile(scores, 100 * (1 - alpha)))


def _finalize_metrics(y_true, scores_test, thr, auc_ci):
    auc_m, auc_lo, auc_hi = auc_ci
    preds = (scores_test >= thr).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "F1": f1_score(y_true, preds, zero_division=0),
        "MCC": matthews_corrcoef(y_true, preds),
        "AUC": auc_m,
        "AUC_CI": f"[{auc_lo:.3f}-{auc_hi:.3f}]",
        "Threshold": thr,
    }, preds


def evaluate_mahalanobis(name, mahal_vn, mahal_va, mahal_test, y_test):
    """Used for the Hybrid model and the Standalone-VAE baseline: the
    anomaly score is the Mahalanobis distance to the nearest component of
    the healthy latent GMM. Threshold via ROC + Youden's Index on
    validation."""
    print(f"\n{'='*60}\nEvaluating: {name}\n{'='*60}")
    thr = find_optimal_threshold(mahal_vn, mahal_va)
    print(f"Threshold T (ROC + Youden's Index) = {thr:.4f}")
    auc_ci = bootstrap_ci(y_test, mahal_test, roc_auc_score)
    metrics, preds = _finalize_metrics(y_test, mahal_test, thr, auc_ci)
    metrics["ThresholdMethod"] = "Youden"
    print(f"{name}: F1={metrics['F1']:.3f} | AUC={metrics['AUC']:.3f} | BalAcc={metrics['Balanced_Accuracy']:.3f}")
    return metrics, mahal_test, preds


def evaluate_single_score(name, scores_vn, scores_va, scores_test, y_test):
    """Used for models with a single scalar anomaly score (Standalone-GAN,
    CBiGAN). Threshold chosen the same way (ROC + Youden's Index) for a
    consistent comparison across all models."""
    print(f"\n{'='*60}\nEvaluating: {name}\n{'='*60}")
    thr = find_optimal_threshold(scores_vn, scores_va)
    print(f"Threshold T (ROC + Youden's Index) = {thr:.4f}")
    auc_ci = bootstrap_ci(y_test, scores_test, roc_auc_score)
    metrics, preds = _finalize_metrics(y_test, scores_test, thr, auc_ci)
    metrics["ThresholdMethod"] = "Youden"
    print(f"{name}: F1={metrics['F1']:.3f} | AUC={metrics['AUC']:.3f} | BalAcc={metrics['Balanced_Accuracy']:.3f}")
    return metrics, scores_test, preds



# ============================================================
## 22. GMM Selection (VALIDATION-ONLY) -- Hybrid & Standalone-VAE
 
def fit_healthy_gmm_sweep(latent_codes, n_components, covariance_type, name="model"):
    gmm = GaussianMixture(
        n_components=n_components, covariance_type=covariance_type,
        random_state=SEED, reg_covar=1e-4,
    ).fit(latent_codes)

    if covariance_type == "full":
        chols = [cho_factor(cov, lower=True) for cov in gmm.covariances_]
        conds = [np.linalg.cond(cov) for cov in gmm.covariances_]
        max_cond = max(conds)
    else:  # 'diag'
        chols = None
        max_cond = float(np.max(gmm.covariances_.max(axis=1) / gmm.covariances_.min(axis=1)))

    print(f"  [{name}] k={n_components} ({covariance_type}) fitted | "
          f"n_samples={latent_codes.shape[0]} | dim={latent_codes.shape[1]} | "
          f"max_cond~={max_cond:.2e} | BIC={gmm.bic(latent_codes):.1f} | AIC={gmm.aic(latent_codes):.1f}")

    return {
        "means": gmm.means_,
        "chols": chols,
        "covariances": gmm.covariances_,
        "covariance_type": covariance_type,
        "n_components": n_components,
        "bic": gmm.bic(latent_codes),
        "aic": gmm.aic(latent_codes),
    }


def mahalanobis_distance_sweep(dist, z):
    dists = []
    if dist["covariance_type"] == "full":
        for mean, chol in zip(dist["means"], dist["chols"]):
            diff = z - mean
            sol = cho_solve(chol, diff.T)
            d2 = np.einsum('ij,ji->i', diff, sol)
            dists.append(np.sqrt(np.maximum(d2, 0.0)))
    else:  # 'diag'
        for mean, var in zip(dist["means"], dist["covariances"]):
            diff = z - mean
            d2 = np.sum((diff ** 2) / var, axis=1)
            dists.append(np.sqrt(np.maximum(d2, 0.0)))
    return np.min(np.stack(dists, axis=0), axis=0)


@torch.no_grad()
def compute_scores_sweep(encoder, loader, dist):
    encoder.eval()
    out = []
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        mu, logvar = encoder(x)
        out.append(mahalanobis_distance_sweep(dist, mu.cpu().numpy()))
    return np.concatenate(out)


def val_auc_for_config(mahal_vn, mahal_va):
    y_val = np.concatenate([np.zeros(len(mahal_vn)), np.ones(len(mahal_va))])
    scores_val = np.concatenate([mahal_vn, mahal_va])
    return roc_auc_score(y_val, scores_val)


sweep_configs = [(2, "full"), (3, "full"), (4, "full"),
                  (2, "diag"), (3, "diag"), (4, "diag")]

print("\n" + "=" * 90)
print("GMM SELECTION -- n_components x covariance_type, validation AUC only")
print("=" * 90)

selection_rows = []
for model_name, encoder, z_healthy in [
    ("Hybrid", hybrid_encoder, z_healthy_hybrid),
    ("Standalone-VAE", standalone_vae.encoder, z_healthy_vae),
]:
    for k, cov_type in sweep_configs:
        dist = fit_healthy_gmm_sweep(z_healthy, k, cov_type, name=model_name)
        mahal_vn = compute_scores_sweep(encoder, loader_val_n, dist)
        mahal_va = compute_scores_sweep(encoder, loader_val_a, dist)
        val_auc = val_auc_for_config(mahal_vn, mahal_va)
        selection_rows.append({
            "Model": model_name, "n_components": k, "covariance_type": cov_type,
            "BIC": dist["bic"], "AIC": dist["aic"], "Val_AUC": val_auc,
        })

df_selection = pd.DataFrame(selection_rows)
df_selection.to_csv(os.path.join(CFG["save_dir"], "gmm_selection_validation.csv"), index=False)
print(df_selection.sort_values(["Model", "Val_AUC"], ascending=[True, False]).to_string(index=False))

best_hybrid_cfg = df_selection[df_selection.Model == "Hybrid"].sort_values("Val_AUC", ascending=False).iloc[0]
best_vae_cfg = df_selection[df_selection.Model == "Standalone-VAE"].sort_values("Val_AUC", ascending=False).iloc[0]
print(f"\nSelected -- Hybrid        : k={int(best_hybrid_cfg.n_components)}, {best_hybrid_cfg.covariance_type} "
      f"(Val_AUC={best_hybrid_cfg.Val_AUC:.4f})")
print(f"Selected -- Standalone-VAE: k={int(best_vae_cfg.n_components)}, {best_vae_cfg.covariance_type} "
      f"(Val_AUC={best_vae_cfg.Val_AUC:.4f})")

# ---- LOCK selection, then touch the test set exactly once per model ----
dist_hybrid_final = fit_healthy_gmm_sweep(
    z_healthy_hybrid, int(best_hybrid_cfg.n_components), best_hybrid_cfg.covariance_type, name="Hybrid-FINAL")
dist_vae_final = fit_healthy_gmm_sweep(
    z_healthy_vae, int(best_vae_cfg.n_components), best_vae_cfg.covariance_type, name="VAE-FINAL")

mahal_vn_h = compute_scores_sweep(hybrid_encoder, loader_val_n, dist_hybrid_final)
mahal_va_h = compute_scores_sweep(hybrid_encoder, loader_val_a, dist_hybrid_final)
mahal_t_h = compute_scores_sweep(hybrid_encoder, loader_test, dist_hybrid_final)
metrics_hybrid, scores_hybrid, preds_hybrid = evaluate_mahalanobis(
    "Hybrid beta-VAE+WGAN-GP (Proposed, GMM validation-tuned)",
    mahal_vn_h, mahal_va_h, mahal_t_h, y_test
)

mahal_vn_v = compute_scores_sweep(standalone_vae.encoder, loader_val_n, dist_vae_final)
mahal_va_v = compute_scores_sweep(standalone_vae.encoder, loader_val_a, dist_vae_final)
mahal_t_v = compute_scores_sweep(standalone_vae.encoder, loader_test, dist_vae_final)
metrics_vae, scores_vae, preds_vae = evaluate_mahalanobis(
    "Standalone beta-VAE (GMM validation-tuned)",
    mahal_vn_v, mahal_va_v, mahal_t_v, y_test
)

# ---- Standalone-GAN: single critic score, no GMM step ----
scores_vn_g = compute_standalone_gan_scores(standalone_gan_C, loader_val_n)
scores_va_g = compute_standalone_gan_scores(standalone_gan_C, loader_val_a)
scores_t_g = compute_standalone_gan_scores(standalone_gan_C, loader_test)
metrics_gan, scores_gan, preds_gan = evaluate_single_score(
    "Standalone WGAN-GP (critic score)",
    scores_vn_g, scores_va_g, scores_t_g, y_test
)



# ============================================================
## 23. Final Results Table (3 models so far -- CBiGAN added in Section 30)
# ============================================================
all_metrics = {
    "Hybrid beta-VAE+WGAN-GP (Proposed)": metrics_hybrid,
    "Standalone beta-VAE": metrics_vae,
    "Standalone WGAN-GP (critic score)": metrics_gan,
}
metric_keys = ["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "Specificity", "F1", "MCC", "AUC"]
df_results = pd.DataFrame(
    {k: {mk: v[mk] for mk in metric_keys} for k, v in all_metrics.items()}
).T.reset_index()
df_results.columns = ["Model"] + metric_keys
df_results["RankScore"] = 0.4 * df_results["AUC"] + 0.3 * df_results["F1"] + 0.3 * df_results["MCC"]
df_results = df_results.sort_values("RankScore", ascending=False).reset_index(drop=True)
df_results.to_csv(os.path.join(CFG["save_dir"], "final_comparison_table.csv"), index=False)

print("\n" + "=" * 70)
print("FINAL RESULTS (3 models)")
print("=" * 70)
print(df_results.to_string(index=False))

# ============================================================
## 24. Visualizations
# ============================================================
def plot_qualitative_comparison(save_path, n=6):
    idx = np.random.choice(len(X_tst_n), n, replace=False)
    x_real = torch.from_numpy(X_tst_n[idx]).permute(0, 3, 1, 2).float().to(device)

    with torch.no_grad():
        mu_h, _ = hybrid_encoder(x_real)
        recon_hybrid = hybrid_generator(mu_h).cpu().permute(0, 2, 3, 1).numpy()
        mu_v, _ = standalone_vae.encoder(x_real)
        recon_vae = standalone_vae.decoder(mu_v).cpu().permute(0, 2, 3, 1).numpy()
        z_rand = torch.randn(n, CFG["latent_dim"], device=device)
        gen_gan = standalone_gan_G(z_rand).cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(4, n, figsize=(2 * n, 8))
    row_titles = ["Real", "Hybrid Recon", "VAE Recon", "GAN Gen"]
    rows = [X_tst_n[idx], recon_hybrid, recon_vae, gen_gan]
    for r in range(4):
        for c in range(n):
            axes[r, c].imshow(np.clip(rows[r][c], 0, 1))
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(row_titles[r], fontsize=10, fontweight="bold")
    plt.suptitle("Qualitative Comparison: Real vs Model Output", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


plot_qualitative_comparison(os.path.join(CFG["save_dir"], "06_qualitative_comparison.png"))

labels_plot = ["Hybrid beta-VAE+WGAN-GP (Proposed)", "Standalone beta-VAE", "Standalone WGAN-GP (critic score)"]
score_sets = [scores_hybrid, scores_vae, scores_gan]
colors_plot = ["#4CAF50", "#2196F3", "#FF9800"]
preds_all = [preds_hybrid, preds_vae, preds_gan]

plt.figure(figsize=(8, 6))
for sc, lab, col in zip(score_sets, labels_plot, colors_plot):
    fpr, tpr, _ = roc_curve(y_test, sc)
    plt.plot(fpr, tpr, label=f"{lab} (AUC={roc_auc_score(y_test, sc):.3f})", color=col, lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curves -- Model Comparison", fontsize=14, fontweight='bold')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(CFG["save_dir"], "07_roc_curves.png"), dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, len(labels_plot), figsize=(6 * len(labels_plot), 5), squeeze=False)
axes = axes[0]
for ax, preds, name in zip(axes, preds_all, labels_plot):
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    ax.set_title(name, fontweight='bold', fontsize=10)
plt.suptitle("Confusion Matrices", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CFG["save_dir"], "08_confusion_matrices.png"), dpi=300, bbox_inches='tight')
plt.show()

metrics_cols = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC"]
x = np.arange(len(metrics_cols))
width = 0.25
fig, ax = plt.subplots(figsize=(13, 6))
for i, (_, row) in enumerate(df_results.iterrows()):
    ax.bar(x + i * width, [row[c] for c in metrics_cols], width, label=row["Model"], alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_cols, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Model Comparison", fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(CFG["save_dir"], "09_metrics_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

print("\nComputing t-SNE of hybrid latent space...")
hybrid_encoder.eval()
latent_vecs = []
with torch.no_grad():
    for batch in loader_test:
        x = batch[0].to(device)
        mu, _ = hybrid_encoder(x)
        latent_vecs.append(mu.cpu().numpy())
latent_vecs = np.concatenate(latent_vecs, axis=0)
latent_vecs = StandardScaler().fit_transform(latent_vecs)
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=SEED).fit_transform(latent_vecs)

plt.figure(figsize=(8, 6))
sc = plt.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=y_test, cmap='coolwarm', alpha=0.7, s=40)
plt.colorbar(sc)
plt.legend(handles=sc.legend_elements()[0], labels=['Normal', 'Abnormal'], fontsize=11)
plt.title("t-SNE Latent Space (Hybrid beta-VAE+WGAN-GP)", fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(CFG["save_dir"], "10_tsne_hybrid.png"), dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
## 25. Final Summary (3 models)
# ============================================================
best_model = df_results.iloc[0]["Model"]
print("\n" + "=" * 70)
print(df_results[["Model", "AUC", "F1", "MCC", "RankScore"]].to_string(index=False))
print("\nBest model (3-way comparison):", best_model)
print("Results directory:", CFG["save_dir"])
print("=" * 70)

# %%
# ============================================================
## 26. Statistical Significance Testing (paired bootstrap + McNemar)
# ============================================================
def paired_bootstrap_test(y_true, score_a, score_b, metric_fn=roc_auc_score,
                           n_bootstrap=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            m_a = metric_fn(yt, score_a[idx])
            m_b = metric_fn(yt, score_b[idx])
        except ValueError:
            continue
        diffs.append(m_a - m_b)
    diffs = np.array(diffs)
    observed = metric_fn(y_true, score_a) - metric_fn(y_true, score_b)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p_value = min(p_value, 1.0)
    return {"observed_diff": float(observed), "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi), "p_value": float(p_value)}


def mcnemar_test(y_true, preds_a, preds_b):
    correct_a = (np.asarray(preds_a) == np.asarray(y_true))
    correct_b = (np.asarray(preds_b) == np.asarray(y_true))
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "no discordant pairs"}
    if n_discordant < 25:
        p_value = stats.binomtest(min(b, c), n_discordant, 0.5).pvalue
        return {"b": b, "c": c, "statistic": None, "p_value": float(p_value), "method": "exact binomial"}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(stat, df=1)
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(p_value),
            "method": "chi-square (continuity corrected)"}


def run_comparison(name_a, score_a, preds_a, name_b, score_b, preds_b, y_true, alpha=0.05):
    print(f"\n{'-'*65}\n{name_a}  vs  {name_b}\n{'-'*65}")
    bt = paired_bootstrap_test(y_true, score_a, score_b, metric_fn=roc_auc_score)
    sig_auc = "significant" if bt["p_value"] < alpha else "not significant"
    print(f"AUC difference   : {bt['observed_diff']:+.4f}  "
          f"[95% CI: {bt['ci_lo']:+.4f}, {bt['ci_hi']:+.4f}]  "
          f"p={bt['p_value']:.4f}  -> {sig_auc}")
    mc = mcnemar_test(y_true, preds_a, preds_b)
    sig_mc = "significant" if mc["p_value"] < alpha else "not significant"
    print(f"McNemar (b={mc['b']}, c={mc['c']}, {mc['method']}): "
          f"p={mc['p_value']:.4f}  -> {sig_mc}")
    return {"Comparison": f"{name_a} vs {name_b}",
            "AUC_diff": bt["observed_diff"], "AUC_CI_lo": bt["ci_lo"], "AUC_CI_hi": bt["ci_hi"],
            "AUC_p_value": bt["p_value"], "McNemar_b": mc["b"], "McNemar_c": mc["c"],
            "McNemar_p_value": mc["p_value"]}


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values, preserving the original order."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=float)

    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, idx in enumerate(order):
        adjusted_value = (m - rank) * p_values[idx]
        running_max = max(running_max, adjusted_value)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


print("\n" + "=" * 70)
print("STATISTICAL SIGNIFICANCE TESTS (alpha = 0.05) -- 3-way comparison")
print("=" * 70)
results_stat = [
    run_comparison("Hybrid (proposed, GMM)", scores_hybrid, preds_hybrid,
                    "Standalone-VAE (GMM)", scores_vae, preds_vae, y_test),
    run_comparison("Hybrid (proposed, GMM)", scores_hybrid, preds_hybrid,
                    "Standalone-GAN (critic score)", scores_gan, preds_gan, y_test),
    run_comparison("Standalone-VAE (GMM)", scores_vae, preds_vae,
                    "Standalone-GAN (critic score)", scores_gan, preds_gan, y_test),
]
df_stat = pd.DataFrame(results_stat)
df_stat.to_csv(os.path.join(CFG["save_dir"], "statistical_tests.csv"), index=False)
print("\n" + "=" * 70)
print("SUMMARY TABLE -- STATISTICAL TESTS (3-way)")
print("=" * 70)
print(df_stat.to_string(index=False))



## 27. Baseline #3 -- CBiGAN (Carrara et al., arXiv:2011.08102)
 
class CBiGANEncoder(nn.Module):
    """Deterministic encoder E(x) -> z (no reparameterization; BiGAN-style
    encoders are deterministic, stochasticity comes from sampling the
    prior z ~ N(0,I) that feeds the generator)."""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU())
        self.res1 = ResidualBlock(64)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.GELU())
        self.res2 = ResidualBlock(128)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.GELU())
        self.res3 = ResidualBlock(256)
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.GELU())
        self.res4 = ResidualBlock(512)
        self.flat_dim = 512 * 14 * 14
        self.fc_z = nn.Linear(self.flat_dim, latent_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.res1(self.enc1(x))
        h = self.res2(self.enc2(h))
        h = self.res3(self.enc3(h))
        h = self.res4(self.enc4(h))
        h = h.reshape(h.size(0), -1)
        return self.fc_z(h).clamp(-50, 50)


class CBiGANJointCritic(nn.Module):
    """Joint discriminator D(x, z). Encodes x through a conv tower to a
    512-d image feature, projects z through an MLP to a 128-d latent
    feature, concatenates and scores the pair. joint_features() exposes
    the pre-head representation, used as f_D(x,z) for the anomaly score."""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.img_net = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, True), nn.Dropout(0.3),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.2, True), nn.Dropout(0.3),
            SelfAttention2d(128),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.2, True), nn.Dropout(0.3),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.InstanceNorm2d(512, affine=True), nn.LeakyReLU(0.2, True), nn.Dropout(0.3),
            spectral_norm(nn.Conv2d(512, 512, 4, 2, 1)),
            nn.InstanceNorm2d(512, affine=True), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.z_net = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 128)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Linear(128, 128)), nn.LeakyReLU(0.2, True),
        )
        self.head = spectral_norm(nn.Linear(512 + 128, 1))

    def joint_features(self, x, z):
        return torch.cat([self.img_net(x), self.z_net(z)], dim=1)

    def forward(self, x, z):
        return self.head(self.joint_features(x, z))


def gradient_penalty_joint(critic, real_pair, fake_pair, device):
    """WGAN-GP on the JOINT (x,z) input space of D: interpolate images and
    latent codes with the same per-sample alpha, penalize deviation of the
    joint gradient norm from 1."""
    x_real, z_real = real_pair
    x_fake, z_fake = fake_pair
    b = x_real.size(0)
    alpha_x = torch.rand(b, 1, 1, 1, device=device)
    alpha_z = alpha_x.view(b, 1)
    x_interp = (alpha_x * x_real + (1 - alpha_x) * x_fake).requires_grad_(True)
    z_interp = (alpha_z * z_real + (1 - alpha_z) * z_fake).requires_grad_(True)
    d_interp = critic(x_interp, z_interp)
    grad_x, grad_z = torch.autograd.grad(
        outputs=d_interp, inputs=[x_interp, z_interp],
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True
    )
    grad_x = grad_x.reshape(b, -1)
    grad_z = grad_z.reshape(b, -1)
    grad_norm = torch.sqrt((grad_x ** 2).sum(1) + (grad_z ** 2).sum(1) + 1e-12)
    return ((grad_norm - 1) ** 2).mean()


def train_cbigan(loader_train, loader_val_n, loader_val_a, cfg):
    """Trains CBiGAN healthy-only, same protocol as the other three
    models. Uses validation-AUC checkpointing + early stopping, mirroring
    the Standalone-WGAN-GP fix."""
    latent_dim = cfg["latent_dim"]
    alpha_consistency = cfg["cbigan_alpha"]
    lam_score = cfg["cbigan_lambda"]
    E = CBiGANEncoder(latent_dim).to(device)
    G = ImprovedGenerator(latent_dim).to(device)
    D = CBiGANJointCritic(latent_dim).to(device)

    opt_EDG = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=cfg["lr"], betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=cfg["lr_critic"], betas=(0.5, 0.9))
    use_amp = torch.cuda.is_available()
    scaler_EG = make_grad_scaler(use_amp)
    scaler_D = make_grad_scaler(use_amp)
    history = {"D": [], "EG": [], "consistency": [], "gp": [], "val_auc": []}

    best_val_auc = -1.0
    best_state = None
    no_improve = 0

    print("\n" + "=" * 70)
    print("BASELINE 3 -- CBiGAN (Carrara et al., arXiv:2011.08102), val-AUC checkpointing")
    print(f"alpha_consistency={alpha_consistency} | lambda_score={lam_score} | "
          f"n_critic={cfg['n_critic']}")
    print("=" * 70)

    @torch.no_grad()
    def _score(loader):
        E.eval(); G.eval(); D.eval()
        scores = []
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            z = E(x)
            x_rec = G(z)
            L_R = (x - x_rec).abs().flatten(1).mean(dim=1)
            f_real = D.joint_features(x, z)
            f_fake = D.joint_features(x_rec, z)
            L_fD = (f_real - f_fake).abs().mean(dim=1)
            A = (1 - lam_score) * L_R + lam_score * L_fD
            scores.append(A.cpu().numpy())
        E.train(); G.train(); D.train()
        return np.concatenate(scores)

    for epoch in range(cfg["epochs"]):
        E.train(); G.train(); D.train()
        ep_D = ep_EG = ep_C = ep_gp = 0.0
        nb_ = 0
        ncs = 0
        for x in tqdm(loader_train, desc=f"CBiGAN {epoch+1}/{cfg['epochs']}", leave=False):
            x = (x[0] if isinstance(x, (list, tuple)) else x).to(device)
            b = x.size(0)

            for _ in range(cfg["n_critic"]):
                z_prior = torch.randn(b, latent_dim, device=device)
                with torch.no_grad():
                    z_enc = E(x)
                    x_gen = G(z_prior)
                with amp_autocast(use_amp):
                    d_real = D(x, z_enc).mean()
                    d_fake = D(x_gen, z_prior).mean()
                gp = gradient_penalty_joint(D, (x.float(), z_enc.float()),
                                             (x_gen.float(), z_prior.float()), device)
                loss_D = -d_real + d_fake + cfg["lambda_gp"] * gp

                opt_D.zero_grad()
                scaler_D.scale(loss_D).backward()
                scaler_D.unscale_(opt_D)
                nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                scaler_D.step(opt_D)
                scaler_D.update()

                ep_D += (d_real - d_fake).item()
                ep_gp += gp.item()
                ncs += 1

            for p in D.parameters():
                p.requires_grad_(False)

            z_prior = torch.randn(b, latent_dim, device=device)
            with amp_autocast(use_amp):
                z_enc = E(x)
                x_gen = G(z_prior)
                d_real = D(x, z_enc).mean()
                d_fake = D(x_gen, z_prior).mean()
                L_EG = d_real - d_fake

                x_cycle = G(z_enc)
                z_cycle = E(x_gen)
                L_C = F.l1_loss(x_cycle, x) + F.l1_loss(z_cycle, z_prior)

                loss_EG = (1 - alpha_consistency) * L_EG + alpha_consistency * L_C

            if torch.isnan(loss_EG) or torch.isinf(loss_EG):
                for p in D.parameters():
                    p.requires_grad_(True)
                continue

            opt_EDG.zero_grad()
            scaler_EG.scale(loss_EG).backward()
            scaler_EG.unscale_(opt_EDG)
            nn.utils.clip_grad_norm_(list(E.parameters()) + list(G.parameters()), 1.0)
            scaler_EG.step(opt_EDG)
            scaler_EG.update()

            for p in D.parameters():
                p.requires_grad_(True)

            ep_EG += L_EG.item()
            ep_C += L_C.item()
            nb_ += 1

        history["D"].append(ep_D / max(ncs, 1))
        history["gp"].append(ep_gp / max(ncs, 1))
        history["EG"].append(ep_EG / max(nb_, 1))
        history["consistency"].append(ep_C / max(nb_, 1))

        s_vn, s_va = _score(loader_val_n), _score(loader_val_a)
        y_val = np.concatenate([np.zeros(len(s_vn)), np.ones(len(s_va))])
        s_val = np.concatenate([s_vn, s_va])
        val_auc = roc_auc_score(y_val, s_val)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch+1:03d} | D(real-fake)={history['D'][-1]:.4f} | "
              f"L_EG={history['EG'][-1]:.4f} | L_C(consistency)={history['consistency'][-1]:.4f} | "
              f"GP={history['gp'][-1]:.4f} | Val_AUC={val_auc:.4f}")

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {
                "E": {k: v.detach().cpu().clone() for k, v in E.state_dict().items()},
                "G": {k: v.detach().cpu().clone() for k, v in G.state_dict().items()},
                "D": {k: v.detach().cpu().clone() for k, v in D.state_dict().items()},
            }
            no_improve = 0
            print(f"  Best CBiGAN checkpoint saved (Val_AUC={val_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"  Early stopping CBiGAN at epoch {epoch+1} "
                      f"(no Val_AUC improvement for {cfg['patience']} epochs)")
                break

    if best_state is not None:
        E.load_state_dict(best_state["E"])
        G.load_state_dict(best_state["G"])
        D.load_state_dict(best_state["D"])
        print(f"Best CBiGAN model restored (Val_AUC={best_val_auc:.4f})")

    torch.save(E.state_dict(), os.path.join(cfg["save_dir"], "cbigan_encoder.pth"))
    torch.save(G.state_dict(), os.path.join(cfg["save_dir"], "cbigan_generator.pth"))
    torch.save(D.state_dict(), os.path.join(cfg["save_dir"], "cbigan_critic.pth"))
    E.eval(); G.eval(); D.eval()
    print("CBiGAN training complete.")
    return E, G, D, history


cleanup_memory()
cbigan_encoder, cbigan_generator, cbigan_critic, cbigan_history = train_cbigan(
    loader_train, loader_val_n, loader_val_a, CFG
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(cbigan_history["EG"], label="L_EG (adversarial)", lw=2, color="#5E35B1")
axes[0].plot(cbigan_history["consistency"], label="L_C (consistency)", lw=2, color="#00897B")
axes[0].set_title("CBiGAN -- Encoder/Generator Losses", fontweight="bold")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(cbigan_history["D"], color="#D81B60", lw=2)
axes[1].set_title("CBiGAN -- Critic D(real) - D(fake)", fontweight="bold")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CFG["save_dir"], "13_cbigan_training_curves.png"), dpi=300, bbox_inches='tight')
plt.show()




# ============================================================
## 28. CBiGAN Anomaly Score + Evaluation
# ============================================================
@torch.no_grad()
def compute_cbigan_scores(E, G, D, loader, lam):
    """A(x) = (1-lambda)*L_R(x) + lambda*L_fD(x)."""
    E.eval(); G.eval(); D.eval()
    scores = []
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        z = E(x)
        x_rec = G(z)
        L_R = (x - x_rec).abs().flatten(1).mean(dim=1)
        f_real = D.joint_features(x, z)
        f_fake = D.joint_features(x_rec, z)
        L_fD = (f_real - f_fake).abs().mean(dim=1)
        A = (1 - lam) * L_R + lam * L_fD
        scores.append(A.cpu().numpy())
    return np.concatenate(scores)


_lam = CFG["cbigan_lambda"]
scores_vn_c = compute_cbigan_scores(cbigan_encoder, cbigan_generator, cbigan_critic, loader_val_n, _lam)
scores_va_c = compute_cbigan_scores(cbigan_encoder, cbigan_generator, cbigan_critic, loader_val_a, _lam)
scores_t_c = compute_cbigan_scores(cbigan_encoder, cbigan_generator, cbigan_critic, loader_test, _lam)
metrics_cbigan, scores_cbigan, preds_cbigan = evaluate_single_score(
    "CBiGAN (Carrara et al., arXiv:2011.08102)",
    scores_vn_c, scores_va_c, scores_t_c, y_test
)

# ============================================================
## 29. Reconstruction-Error Comparison (Normal vs Abnormal, per model)
#     Diagnostic only -- the actual anomaly scores used for AUC/F1/etc
#     are the latent-space GMM-Mahalanobis distance (Hybrid, Standalone-
#     VAE), the critic score (Standalone-GAN), and the CBiGAN A(x) mix;
#     none of them is raw reconstruction error alone.
# ============================================================
@torch.no_grad()
def per_sample_recon_error(encode_fn, decode_fn, images, batch_size=32):
    loader = DataLoader(CellDataset(images), batch_size=batch_size, shuffle=False)
    errs = []
    for x in loader:
        x = x.to(device)
        z = encode_fn(x)
        x_rec = decode_fn(z)
        e = (x - x_rec).abs().flatten(1).mean(dim=1)
        errs.append(e.cpu().numpy())
    return np.concatenate(errs)


hybrid_encoder.eval(); hybrid_generator.eval()
standalone_vae.eval()
cbigan_encoder.eval(); cbigan_generator.eval()

rec_err_normal, rec_err_abnormal = {}, {}

rec_err_normal["Hybrid beta-VAE+WGAN-GP"] = per_sample_recon_error(
    lambda x: hybrid_encoder(x)[0], hybrid_generator, X_tst_n)
rec_err_abnormal["Hybrid beta-VAE+WGAN-GP"] = per_sample_recon_error(
    lambda x: hybrid_encoder(x)[0], hybrid_generator, X_tst_a)

rec_err_normal["Standalone beta-VAE"] = per_sample_recon_error(
    lambda x: standalone_vae.encoder(x)[0], standalone_vae.decoder, X_tst_n)
rec_err_abnormal["Standalone beta-VAE"] = per_sample_recon_error(
    lambda x: standalone_vae.encoder(x)[0], standalone_vae.decoder, X_tst_a)

rec_err_normal["CBiGAN"] = per_sample_recon_error(cbigan_encoder, cbigan_generator, X_tst_n)
rec_err_abnormal["CBiGAN"] = per_sample_recon_error(cbigan_encoder, cbigan_generator, X_tst_a)

print("\nNote: Standalone-GAN has no encoder (only prior-sampled generation), "
      "so it has no defined per-sample reconstruction error and is excluded "
      "from this specific plot (it IS included in all other comparisons).")

rows = []
for name in rec_err_normal:
    for v in rec_err_normal[name]:
        rows.append({"Model": name, "Class": "Normal", "ReconError": float(v)})
    for v in rec_err_abnormal[name]:
        rows.append({"Model": name, "Class": "Abnormal", "ReconError": float(v)})
df_recon = pd.DataFrame(rows)
df_recon.to_csv(os.path.join(CFG["save_dir"], "reconstruction_errors_per_sample.csv"), index=False)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_recon, x="Model", y="ReconError", hue="Class",
            palette={"Normal": "#2196F3", "Abnormal": "#F44336"}, showfliers=False)
sns.stripplot(data=df_recon, x="Model", y="ReconError", hue="Class", dodge=True,
              palette={"Normal": "#0D47A1", "Abnormal": "#B71C1C"},
              alpha=0.25, size=2, legend=False)
plt.title("Per-Sample Pixel Reconstruction Error (L1) -- Normal vs Abnormal, by Model",
          fontsize=13, fontweight='bold')
plt.ylabel("Mean |x - G(E(x))| (pixel L1)")
plt.xlabel("")
plt.xticks(rotation=8)
handles, labels_ = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels_[:2], title="Test-set class")
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(CFG["save_dir"], "14_reconstruction_error_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

print("\nReconstruction-error separation summary (raw pixel L1, informational only):")
for name in rec_err_normal:
    en, ea = rec_err_normal[name], rec_err_abnormal[name]
    auc_recon = roc_auc_score(
        np.concatenate([np.zeros(len(en)), np.ones(len(ea))]),
        np.concatenate([en, ea])
    )
    print(f"  {name:32s}: mean(Normal)={en.mean():.4f}  mean(Abnormal)={ea.mean():.4f}  "
          f"AUC(raw recon error alone)={auc_recon:.3f}")

# ============================================================
## 30. Extended Final Comparison -- ALL 4 MODELS (incl. CBiGAN)
# ============================================================
all_metrics["CBiGAN (Carrara et al., arXiv:2011.08102)"] = metrics_cbigan
df_results_ext = pd.DataFrame(
    {k: {mk: v[mk] for mk in metric_keys} for k, v in all_metrics.items()}
).T.reset_index()
df_results_ext.columns = ["Model"] + metric_keys
df_results_ext["RankScore"] = 0.4 * df_results_ext["AUC"] + 0.3 * df_results_ext["F1"] + 0.3 * df_results_ext["MCC"]
df_results_ext = df_results_ext.sort_values("RankScore", ascending=False).reset_index(drop=True)
df_results_ext.to_csv(os.path.join(CFG["save_dir"], "final_comparison_table_with_cbigan.csv"), index=False)

print("\n" + "=" * 70)
print("FINAL RESULTS -- ALL 4 MODELS (incl. CBiGAN baseline)")
print("=" * 70)
print(df_results_ext.to_string(index=False))

labels_plot_ext = labels_plot + ["CBiGAN (Carrara et al., arXiv:2011.08102)"]
score_sets_ext = score_sets + [scores_cbigan]
colors_plot_ext = colors_plot + ["#9C27B0"]
preds_all_ext = preds_all + [preds_cbigan]

plt.figure(figsize=(8, 6))
for sc, lab, col in zip(score_sets_ext, labels_plot_ext, colors_plot_ext):
    fpr, tpr, _ = roc_curve(y_test, sc)
    plt.plot(fpr, tpr, label=f"{lab} (AUC={roc_auc_score(y_test, sc):.3f})", color=col, lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curves -- Including CBiGAN Baseline", fontsize=14, fontweight='bold')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(CFG["save_dir"], "15_roc_curves_with_cbigan.png"), dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, len(labels_plot_ext), figsize=(6 * len(labels_plot_ext), 5), squeeze=False)
axes = axes[0]
for ax, preds, name in zip(axes, preds_all_ext, labels_plot_ext):
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    ax.set_title(name, fontweight='bold', fontsize=9)
plt.suptitle("Confusion Matrices -- Including CBiGAN Baseline", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CFG["save_dir"], "16_confusion_matrices_with_cbigan.png"), dpi=300, bbox_inches='tight')
plt.show()

metrics_cols = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC"]
x = np.arange(len(metrics_cols))
width = 0.8 / len(df_results_ext)
fig, ax = plt.subplots(figsize=(14, 6))
for i, (_, row) in enumerate(df_results_ext.iterrows()):
    ax.bar(x + i * width, [row[c] for c in metrics_cols], width, label=row["Model"], alpha=0.85)
ax.set_xticks(x + width * (len(df_results_ext) - 1) / 2)
ax.set_xticklabels(metrics_cols, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Model Comparison -- Including CBiGAN Baseline", fontsize=14, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(CFG["save_dir"], "17_metrics_comparison_with_cbigan.png"), dpi=300, bbox_inches='tight')
plt.show()


def plot_qualitative_with_cbigan(save_path, n=6):
    idx = np.random.choice(len(X_tst_n), n, replace=False)
    x_real = torch.from_numpy(X_tst_n[idx]).permute(0, 3, 1, 2).float().to(device)
    with torch.no_grad():
        mu_h, _ = hybrid_encoder(x_real)
        recon_hybrid = hybrid_generator(mu_h).cpu().permute(0, 2, 3, 1).numpy()
        mu_v, _ = standalone_vae.encoder(x_real)
        recon_vae = standalone_vae.decoder(mu_v).cpu().permute(0, 2, 3, 1).numpy()
        z_c = cbigan_encoder(x_real)
        recon_cbigan = cbigan_generator(z_c).cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(4, n, figsize=(2 * n, 8))
    row_titles = ["Real", "Hybrid Recon", "VAE Recon", "CBiGAN Recon"]
    rows_ = [X_tst_n[idx], recon_hybrid, recon_vae, recon_cbigan]
    for r in range(4):
        for c in range(n):
            axes[r, c].imshow(np.clip(rows_[r][c], 0, 1))
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(row_titles[r], fontsize=10, fontweight="bold")
    plt.suptitle("Qualitative Comparison: Real vs Reconstructions (incl. CBiGAN)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


plot_qualitative_with_cbigan(os.path.join(CFG["save_dir"], "18_qualitative_comparison_with_cbigan.png"))



# ============================================================
## 31. Statistical Significance -- CBiGAN vs the Other 3 Models
# ============================================================
print("\n" + "=" * 70)
print("STATISTICAL SIGNIFICANCE TESTS -- CBiGAN vs other models (alpha=0.05)")
print("=" * 70)

results_stat_cbigan = [
    run_comparison("Hybrid (proposed, GMM)", scores_hybrid, preds_hybrid,
                    "CBiGAN (Carrara et al.)", scores_cbigan, preds_cbigan, y_test),
    run_comparison("Standalone-VAE (GMM)", scores_vae, preds_vae,
                    "CBiGAN (Carrara et al.)", scores_cbigan, preds_cbigan, y_test),
    run_comparison("Standalone-GAN (critic score)", scores_gan, preds_gan,
                    "CBiGAN (Carrara et al.)", scores_cbigan, preds_cbigan, y_test),
]
df_stat_cbigan = pd.DataFrame(results_stat_cbigan)

# Holm-Bonferroni correction across all six pairwise AUC comparisons.
df_stat_all = pd.concat([df_stat, df_stat_cbigan], ignore_index=True)
df_stat_all["AUC_p_holm"] = holm_adjust(df_stat_all["AUC_p_value"].to_numpy())
df_stat_all["AUC_significant_holm"] = df_stat_all["AUC_p_holm"] < 0.05

# Restore the two original three-comparison tables, now with the global Holm correction.
df_stat = df_stat_all.iloc[:len(df_stat)].copy()
df_stat_cbigan = df_stat_all.iloc[len(df_stat):].copy()

df_stat.to_csv(os.path.join(CFG["save_dir"], "statistical_tests.csv"), index=False)
df_stat_cbigan.to_csv(os.path.join(CFG["save_dir"], "statistical_tests_with_cbigan.csv"), index=False)
df_stat_all.to_csv(os.path.join(CFG["save_dir"], "statistical_tests_all_6_pairwise_with_holm.csv"), index=False)

print("\n" + "=" * 70)
print("SUMMARY TABLE -- ALL SIX AUC COMPARISONS (Holm-Bonferroni corrected)")
print("=" * 70)
print(df_stat_all.to_string(index=False))

print("\n" + "=" * 70)
print("ALL OUTPUTS SAVED TO:", CFG["save_dir"])
print("=" * 70)
print("\nFINAL 4-MODEL COMPARISON:")
print(df_results_ext[["Model", "AUC", "F1", "MCC", "Balanced_Accuracy", "RankScore"]].to_string(index=False))
print("\nBest model (4-way comparison):", df_results_ext.iloc[0]["Model"])


