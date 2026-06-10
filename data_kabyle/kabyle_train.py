#!/usr/bin/env python3
# train_kabyle.py

import os
import sys
import subprocess
import yaml
import torch
from pathlib import Path

# ---------- CONFIGURATION ----------
PROJECT_ROOT = Path.cwd()
FONTS_DIR = PROJECT_ROOT / "fonts"
DICT_FILE = PROJECT_ROOT / "kab.txt"
CHAR_FILE = PROJECT_ROOT / "kab.char.txt"
NUM_SYNTH_IMAGES = 10000        # Nombre d'images synthétiques à générer
BATCH_SIZE = 64                 # À réduire si votre GPU manque de mémoire (ex: 32, 16)
NUM_ITER = 5000                 # Nombre d'itérations d'entraînement
MODEL_NAME = "kab_model"        # Ce nom doit être identique pour les 3 fichiers

# Chemins d'installation d'EasyOCR (sous Linux/Mac)
EASYOCR_HOME = Path.home() / ".EasyOCR"
MODEL_DIR = EASYOCR_HOME / "model"
USER_NETWORK_DIR = EASYOCR_HOME / "user_network"
# -----------------------------------

# Détection CPU/GPU
USE_GPU = torch.cuda.is_available()
DEVICE = "GPU" if USE_GPU else "CPU"
print(f"[INFO] Périphérique détecté : {DEVICE}")

# Vérification des fichiers obligatoires
for f in [DICT_FILE, CHAR_FILE]:
    if not f.exists():
        print(f"[ERREUR] Fichier manquant : {f}")
        sys.exit(1)
if not FONTS_DIR.is_dir() or not any(f.suffix.lower() in [".ttf", ".otf"] for f in FONTS_DIR.iterdir()):
    print(f"[ERREUR] Dossier fonts invalide ou vide : {FONTS_DIR}")
    sys.exit(1)

# ---------- FONCTIONS UTILITAIRES ----------
def run_cmd(cmd, cwd=None, check=True):
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=check)

# ---------- 1. CLONAGE DES DÉPÔTS ----------
def clone_repos():
    repos = [
        ("deep-text-recognition-benchmark", "https://github.com/clovaai/deep-text-recognition-benchmark.git"),
        ("TextRecognitionDataGenerator", "https://github.com/Belval/TextRecognitionDataGenerator.git")
    ]
    for name, url in repos:
        if not Path(name).exists():
            run_cmd(["git", "clone", url])

# ---------- 2. INSTALLATION DES DÉPENDANCES ----------
def install_deps():
    packages = ["torch", "torchvision", "lmdb", "natsort", "opencv-python", "fire", "nltk", "PyYAML"]
    for pkg in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

# ---------- 3. GÉNÉRATION DES DONNÉES SYNTHÉTIQUES ----------
def generate_synthetic_data():
    out_dir = PROJECT_ROOT / "synth_output"
    run_cmd([
        "python", "TextRecognitionDataGenerator/run.py",
        "-w", "1",
        "-c", str(NUM_SYNTH_IMAGES),
        "-f", str(FONTS_DIR),
        "--dict", str(DICT_FILE),
        "--output_dir", str(out_dir)
    ])
    # Création du fichier labels.txt
    labels_file = out_dir / "labels.txt"
    with open(labels_file, "w") as f:
        for img in out_dir.glob("*.jpg"):
            text = img.stem.split("_")[0]
            f.write(f"{img}\t{text}\n")
    return out_dir

# ---------- 4. CONVERSION VERS LMDB ----------
def convert_to_lmdb(data_dir):
    lmdb_path = PROJECT_ROOT / "lmdb_data"
    run_cmd([
        "python", "deep-text-recognition-benchmark/create_lmdb_dataset.py",
        "--inputPath", str(data_dir),
        "--gtFile", str(data_dir / "labels.txt"),
        "--outputPath", str(lmdb_path)
    ])
    return lmdb_path

# ---------- 5. ENTRAÎNEMENT ----------
def train_model(lmdb_path):
    cmd = [
        "python", "deep-text-recognition-benchmark/train.py",
        "--train_data", str(lmdb_path),
        "--valid_data", str(lmdb_path),
        "--select_data", "/",
        "--batch_ratio", "1.0",
        "--Transformation", "TPS",
        "--FeatureExtraction", "ResNet",
        "--SequenceModeling", "BiLSTM",
        "--Prediction", "Attn",
        "--batch_size", str(BATCH_SIZE),
        "--data_filtering_off",
        "--workers", "0",
        "--batch_max_length", "80",
        "--num_iter", str(NUM_ITER),
        "--valInterval", "500",
        "--saved_model", "TPS-ResNet-BiLSTM-Attn.pth",
        "--adam",
        "--lr", "1e-3",
        "--experiment", MODEL_NAME
    ]
    if USE_GPU:
        cmd.append("--cuda")
    else:
        # Forcer le CPU pour charger le modèle pré-entraîné
        train_script = Path("deep-text-recognition-benchmark/train.py")
        content = train_script.read_text()
        content = content.replace(
            "model.load_state_dict(torch.load(opt.saved_model))",
            "model.load_state_dict(torch.load(opt.saved_model, map_location='cpu'))"
        )
        train_script.write_text(content)

    run_cmd(cmd)

# ---------- 6. GÉNÉRATION DES FICHIERS .yaml ET .py ----------
def generate_easyocr_files():
    # 1. Lecture du fichier des caractères
    with open(CHAR_FILE, "r") as f:
        characters = f.read().strip()
        # S'assurer que les caractères sont au bon format
        characters = ''.join(sorted(set(characters)))
    print(f"[INFO] Jeu de caractères : {characters[:50]}...")

    # 2. Création du fichier YAML
    yaml_content = {
        "imgH": 32,
        "lang_list": ["kab"],  # Code ISO 639-1 pour le kabyle
        "network_params": {
            "input_channel": 1,
            "output_channel": 512,
            "hidden_size": 256
        },
        "character_list": characters
    }
    yaml_path = PROJECT_ROOT / f"{MODEL_NAME}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Fichier YAML généré : {yaml_path}")

    # 3. Création du fichier PY (architecture de base)
    py_content = '''
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )
        self.SequenceModeling_output = hidden_size
        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input, text):
        # Feature extraction stage
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)
        # Sequence modeling stage
        contextual_feature = self.SequenceModeling(visual_feature)
        # Prediction stage
        prediction = self.Prediction(contextual_feature.contiguous())
        return prediction

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        try:
            self.rnn.flatten_parameters()
        except:
            pass
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)
        return output

class VGG_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True)
        )

    def forward(self, input):
        return self.ConvNet(input)
'''
    py_path = PROJECT_ROOT / f"{MODEL_NAME}.py"
    py_path.write_text(py_content.strip())
    print(f"[INFO] Fichier PY généré : {py_path}")

# ---------- 7. DÉPLOIEMENT DANS LES DOSSIERS EASYOCR ----------
def deploy_easyocr_files():
    # Création des répertoires s'ils n'existent pas
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    USER_NETWORK_DIR.mkdir(parents=True, exist_ok=True)

    # Déplacement/copie des fichiers
    pth_src = PROJECT_ROOT / f"{MODEL_NAME}" / "best_accuracy.pth"
    if pth_src.exists():
        pth_dest = MODEL_DIR / f"{MODEL_NAME}.pth"
        import shutil
        shutil.copy(pth_src, pth_dest)
        print(f"[INFO] Modèle .pth déployé vers : {pth_dest}")
    else:
        print(f"[AVERTISSEMENT] Fichier .pth introuvable : {pth_src}")

    for ext in [".yaml", ".py"]:
        src = PROJECT_ROOT / f"{MODEL_NAME}{ext}"
        if src.exists():
            dest = USER_NETWORK_DIR / src.name
            import shutil
            shutil.copy(src, dest)
            print(f"[INFO] Fichier {ext} déployé vers : {dest}")
        else:
            print(f"[AVERTISSEMENT] Fichier {ext} introuvable : {src}")

# ---------- MAIN ----------
def main():
    clone_repos()
    install_deps()
    synth_dir = generate_synthetic_data()
    lmdb_dir = convert_to_lmdb(synth_dir)
    train_model(lmdb_dir)
    generate_easyocr_files()
    deploy_easyocr_files()
    print(f"\n[SUCCÈS] Entraînement terminé. Votre modèle '{MODEL_NAME}' est prêt !")
    print(f"Pour l'utiliser : reader = easyocr.Reader(['kab'], recog_network='{MODEL_NAME}')")

if __name__ == "__main__":
    main()