import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DenseNet121(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        for param in base_model.features.parameters():
            param.requires_grad = False

        self.features = base_model.features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),  
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.scaler = GradScaler()

    def forward(self, x):
        B, T, H, W = x.shape
        x = x.unsqueeze(2).view(B * T, 1, H, W).repeat(1, 3, 1, 1)
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        x = x.view(B, T, -1).mean(dim=1)
        return x

    def train_model(self, train_loader, val_loader, idx2lbl, device, *, epochs=25, lr=1e-5):
        self.to(device)
        label_counts = Counter(lbl for _, lbl in train_loader.dataset.data)
        total = sum(label_counts.values())
        weights = torch.tensor([total / label_counts[i] for i in range(len(idx2lbl))], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.AdamW(self.classifier.parameters(), lr=lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

        print("\n📊 Balanceamento no treino:")
        for idx in sorted(label_counts):
            print(f" - {idx2lbl[idx]}: {label_counts[idx]} exemplo(s)")

        for ep in range(1, epochs + 1):
            self.train()
            loss_sum = corr = total = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                # Usando mixed precision
                with autocast():
                    out = self(x)
                    loss = criterion(out, y)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                loss_sum += loss.item()
                preds = out.argmax(1)
                corr += (preds == y).sum().item()
                total += y.size(0)

            print(f"📘 Epoch {ep:2d}/{epochs} | Loss: {loss_sum/len(train_loader):.4f} | Acc: {100*corr/total:.2f}%")
            scheduler.step(loss_sum/len(train_loader))

        return self.evaluate(val_loader, idx2lbl, device, "validação")

    def evaluate(self, loader, idx2lbl, device, dataset_name="teste"):
        self.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = self(x).argmax(1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
        return self._report(y_true, y_pred, idx2lbl, dataset_name)

    def _report(self, y_true, y_pred, idx2lbl, name):
        total = len(y_true)
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        acc = 100 * correct / total if total else 0

        report = [
            "=" * 70,
            f"🧪 RELATÓRIO ({name.upper()})",
            "=" * 70,
            f"Total de amostras : {total}",
            f"Acertos            : {correct}",
            f"Erros              : {total - correct}",
            f"Acurácia geral     : {acc:.2f}%",
            "-" * 70,
            "📊 Acurácia por letra:"
        ]

        per_tot = Counter(y_true)
        per_hit = Counter([t for t, p in zip(y_true, y_pred) if t == p])
        for idx in sorted(idx2lbl.keys()):
            total_l = per_tot.get(idx, 0)
            hits_l = per_hit.get(idx, 0)
            acc_l = 100 * hits_l / total_l if total_l else 0
            report.append(f"   {idx2lbl[idx]} : {acc_l:.2f}% ({hits_l}/{total_l})")

        # Cálculo das métricas globais (macro)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall    = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1        = f1_score(y_true, y_pred, average='macro', zero_division=0)

        report.append("-" * 70)
        report.append(f"Precision (macro)   : {precision:.4f}")
        report.append(f"Recall (macro)  : {recall:.4f}")
        report.append(f"F1-score (macro)   : {f1:.4f}")
        report.append("=" * 70)

        # Matriz de confusão (texto)
        cm = confusion_matrix(y_true, y_pred, labels=list(idx2lbl.keys()))
        report.append("🧩 Matriz de confusão (linhas = verdadeiro, colunas = predito):")
        report.append(np.array2string(cm, separator=', '))

        # Matriz de confusão (imagem)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[idx2lbl[i] for i in sorted(idx2lbl)],
                    yticklabels=[idx2lbl[i] for i in sorted(idx2lbl)])
        ax.set_xlabel('Previsto')
        ax.set_ylabel('Verdadeiro')
        ax.set_title(f'Matriz de Confusão - {name}')
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_densenet_{name.lower()}.png")
        plt.close()

        print("\n".join(report))
        with open("relatorio_resultado_densenet.txt", "a") as f:
            f.write("\n".join(report) + "\n\n")

        return acc

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"✅ Modelo salvo em: {path}")
