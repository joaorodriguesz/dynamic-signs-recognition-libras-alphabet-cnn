import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Preprocessor:
    def __init__(
        self,
        num_frames: int      = 20,
        frame_size: tuple    = (224, 224),
        save_dir:   str      = "../data/numpy_data",
        augment:    bool     = False,
        augment_factor: int  = 10
    ):
        self.num_frames     = num_frames
        self.frame_size     = frame_size
        self.save_dir       = save_dir
        self.augment        = augment
        self.augment_factor = augment_factor
        os.makedirs(save_dir, exist_ok=True)

    def load_video(self, video_path: str):
        cap, frames = cv2.VideoCapture(video_path), []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            frames.append(frame)
        cap.release()
        return frames

    @staticmethod
    def apply_filter(frames):
        """Desfocagem Gaussiana leve para redução de ruído."""
        return [cv2.GaussianBlur(f, (5, 5), 0) for f in frames]

    def resize_frames(self, frames):
        return [cv2.resize(f, self.frame_size) for f in frames]

    @staticmethod
    def normalize_frames(frames):
        return np.array(frames, dtype=np.float32) / 255.0  

    def fixed_augmentation(self, frames, aug_index):

        def scharr_edge_detection(image):
            """Aplica o filtro Scharr para detecção de bordas na imagem."""
            # Aplicando Scharr para detecção de bordas
            grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
            grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
            grad = cv2.magnitude(grad_x, grad_y)
            
            return cv2.convertScaleAbs(grad)
        
        f = frames.copy()
        h, w = self.frame_size

        if aug_index == 0:
            # Ruído gaussiano com mais intensidade (agora no índice 4)
            std = 30 
            f = [np.clip(frame + np.random.normal(0, std, frame.shape), 0, 255).astype(np.uint8) for frame in f]

        elif aug_index == 1:
            # Espelhamento horizontal + leve sharpening
            f = [cv2.flip(frame, 1) for frame in f]
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            f = [cv2.filter2D(frame, -1, kernel) for frame in f]

        elif aug_index == 2:
            # Combinação de aumento de contraste e ruído gaussiano
            std = 20  
            f = [np.clip(frame + np.random.normal(0, std, frame.shape), 0, 255).astype(np.uint8) for frame in f]
            f = [cv2.convertScaleAbs(frame, alpha=3.0, beta=0) for frame in f]  # Aumento de contraste

        elif aug_index == 3:
            # Ruído gaussiano com espelhamento horizontal
            std = 30  # Aumento do ruído
            f = [np.clip(frame + np.random.normal(0, std, frame.shape), 0, 255).astype(np.uint8) for frame in f]
            f = [cv2.flip(frame, 1) for frame in f]  

        elif aug_index == 4:
            # Aumento de brilho e contraste + espelhamento
            brightness = 60
            f = [cv2.convertScaleAbs(frame, alpha=2.0, beta=brightness) for frame in f]
            f = [cv2.flip(frame, 1) for frame in f]  

        elif aug_index == 5:
            # Aplicando Gaussian Blur para suavizar a imagem e depois a detecção de bordas (Scharr)
            f = [cv2.GaussianBlur(frame, (5, 5), 0) for frame in f]  
            f = [scharr_edge_detection(frame) for frame in f]  

        elif aug_index == 6:
            # Aumento de contraste sem rotação ou zoom
            f = [cv2.convertScaleAbs(frame, alpha=3.0, beta=0) for frame in f] 

        elif aug_index == 7:
            # Aplicando o filtro de Scharr no índice 6 e espelhando a imagem
            f = [cv2.GaussianBlur(frame, (5, 5), 0) for frame in f]  
            f = [scharr_edge_detection(frame) for frame in f]  
            f = [cv2.flip(frame, 1) for frame in f] 

        elif aug_index == 8:
            # Aplicando o filtro de ruído gaussiano e aumento de contraste e espelhando
            std = 20  
            f = [np.clip(frame + np.random.normal(0, std, frame.shape), 0, 255).astype(np.uint8) for frame in f]
            f = [cv2.convertScaleAbs(frame, alpha=3.0, beta=0) for frame in f]  
            f = [cv2.flip(frame, 1) for frame in f]  

        elif aug_index == 9:
            # Original
            pass


        return f

    def select_frames(self, frames):
        tot = len(frames)
        if tot < self.num_frames:                   # pad com frames zerados
            frames += [np.zeros_like(frames[0])] * (self.num_frames - tot)
        else:                                       # amostragem uniforme
            step   = tot // self.num_frames
            frames = [frames[i * step] for i in range(self.num_frames)]
        return np.array(frames)

    def process_and_save(self, video_path: str, video_name: str, label: str):
        base = self.apply_filter(self.load_video(video_path))
        reps = self.augment_factor if self.augment else 1

        for i in range(reps):
            if self.augment:
                fr = self.fixed_augmentation(base.copy(), i)
            else:
                fr = base.copy()


            fr = self.resize_frames(fr)
            fr = self.normalize_frames(fr)
            fr = self.select_frames(fr)

            lbl_dir = os.path.join(self.save_dir, label)
            os.makedirs(lbl_dir, exist_ok=True)
            fname = f"{video_name}_aug{i}" if self.augment else video_name
            np.save(os.path.join(lbl_dir, f"{fname}.npy"), fr)

        print(f"✅ {video_name}: {reps} arquivo(s) salvo(s)")

    def _collect_samples(self, data_dir):
        """Varre data_dir e devolve (paths, labels, idx→label)."""
        samples, labels, idx2lbl = [], [], {}
        for idx, lbl in enumerate(sorted(os.listdir(data_dir))):
            idx2lbl[idx] = lbl
            lbl_path = os.path.join(data_dir, lbl)
            for f in os.listdir(lbl_path):
                if f.endswith(".npy"):
                    samples.append(os.path.join(lbl_path, f))
                    labels.append(idx)
        return samples, labels, idx2lbl

    def _make_loader(self, data_tuples, batch_size, shuffle):
        """data_tuples = [(path, label), ...]"""
        class _DS(Dataset):
            def __init__(self, data):
                self.data = data    

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                path, lbl = self.data[idx]
                x   = np.load(path)
                x_t = torch.tensor(x,  dtype=torch.float32)
                y_t = torch.tensor(lbl, dtype=torch.long)
                return x_t, y_t

        return DataLoader(
            _DS(data_tuples),
            batch_size=batch_size,
            shuffle=shuffle
        )