import os
import shutil
from sklearn.model_selection import train_test_split
from preprocess.preprocessor   import Preprocessor
from preprocess.video_manager  import VideoManager

RAW_DIR = (
    "/home/joaorodriguesz/dev/workspace/github/"
    "dynamic-signs-recognition-libras-alphabet-cnn/data/raw_videos"
)
NPY_DIR = (
    "/home/joaorodriguesz/dev/workspace/github/"
    "dynamic-signs-recognition-libras-alphabet-cnn/data/numpy_data"
)
SPLIT_DIR = (
    "/home/joaorodriguesz/dev/workspace/github/"
    "dynamic-signs-recognition-libras-alphabet-cnn/data/split"
)

video_manager = VideoManager(RAW_DIR)
preprocessor  = Preprocessor(
    save_dir       = NPY_DIR,
    augment        = True,   
    augment_factor = 10       
)

DATASETS = ["vlibras", "ines_gov"]

print("\n🚀 Iniciando pré‑processamento + data‑augmentation...\n")

for dataset in DATASETS:
    ds_path = os.path.join(RAW_DIR, dataset)
    if not os.path.isdir(ds_path):
        print(f"⚠️  Diretório {ds_path} não encontrado. Pulando...")
        continue

    for letter in sorted(os.listdir(ds_path)):
        letter_path = os.path.join(ds_path, letter)
        if not os.path.isdir(letter_path):
            continue

        print(f"🔄  {dataset.upper()} | Letra {letter.upper()}")
        videos = video_manager.list_videos(letter_path)
        if not videos:
            print("   (nenhum vídeo encontrado)\n")
            continue

        for v in videos:
            vid_name = os.path.splitext(os.path.basename(v))[0]
            preprocessor.process_and_save(v, vid_name, letter.upper())

print("\n✅ Pré‑processamento concluído!\n")

total = sum(
    len([f for f in os.listdir(os.path.join(NPY_DIR, l)) if f.endswith(".npy")])
    for l in os.listdir(NPY_DIR)
    if os.path.isdir(os.path.join(NPY_DIR, l))
)
print(f"🔍 Total de amostras geradas (com aug.): {total}")

print("\n📦 Gerando e salvando splits fixos (train / val / test)...")

for split in ["train", "val", "test"]:
    split_path = os.path.join(SPLIT_DIR, split)
    if os.path.exists(split_path):
        shutil.rmtree(split_path)
    os.makedirs(split_path)

samples = []
for label in sorted(os.listdir(NPY_DIR)):
    label_dir = os.path.join(NPY_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    for f in os.listdir(label_dir):
        if f.endswith(".npy"):
            samples.append((os.path.join(label_dir, f), label))

paths, labels = zip(*samples)
train_val, test = train_test_split(samples, test_size=0.15, stratify=labels, random_state=42)
train, val = train_test_split(train_val, test_size=0.15 / (1 - 0.15), stratify=[l for _, l in train_val], random_state=42)

splits = {"train": train, "val": val, "test": test}

for split_name, data in splits.items():
    for src, label in data:
        dest_dir = os.path.join(SPLIT_DIR, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(src))
        shutil.copy2(src, dest_path)

print("✅ Splits fixos salvos em: data/split")
print("\n🎯 Tamanhos dos splits:")
print(f"   • Train: {len(train)} amostras")
print(f"   • Val  : {len(val)} amostras")
print(f"   • Test : {len(test)} amostras")


