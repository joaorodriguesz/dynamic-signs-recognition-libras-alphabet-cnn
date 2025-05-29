import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import torchvision.models as models
from torch.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class InceptionV3Net(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)

        # Liberar apenas as últimas camadas
        for name, param in base_model.named_parameters():
            if "Mixed_7" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.features = base_model
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.scaler = GradScaler()

    def forward(self, x):
        B, T, H, W = x.shape
        x = x.view(B * T, 1, H, W).expand(-1, 3, -1, -1)

        # Passa pelo modelo até o Mixed_7c (última camada liberada)
        x = self.features.Conv2d_1a_3x3(x)
        x = self.features.Conv2d_2a_3x3(x)
        x = self.features.Conv2d_2b_3x3(x)
        x = self.features.maxpool1(x)
        x = self.features.Conv2d_3b_1x1(x)
        x = self.features.Conv2d_4a_3x3(x)
        x = self.features.maxpool2(x)
        x = self.features.Mixed_5b(x)
        x = self.features.Mixed_5c(x)
        x = self.features.Mixed_5d(x)
        x = self.features.Mixed_6a(x)
        x = self.features.Mixed_6b(x)
        x = self.features.Mixed_6c(x)
        x = self.features.Mixed_6d(x)
        x = self.features.Mixed_6e(x)
        x = self.features.Mixed_7a(x)
        x = self.features.Mixed_7b(x)
        x = self.features.Mixed_7c(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = x.view(B, T, -1).mean(dim=1)
        return x

    def train_model(self, train_loader, val_loader, idx2lbl, device, *, epochs=25, lr=1e-4):
        self.to(device)
        label_counts = Counter(lbl for _, lbl in train_loader.dataset.data)
        total = sum(label_counts.values())
        weights = torch.tensor(
            [total / label_counts[i] for i in range(len(idx2lbl))],
            dtype=torch.float32, device=device
        )
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Otimizador com lr menor pro backbone
        optimizer = optim.AdamW([
            {'params': filter(lambda p: p.requires_grad, self.features.parameters()), 'lr': 1e-5},
            {'params': self.classifier.parameters(), 'lr': lr}
        ])

        print("\n📊 Balanceamento no treino:")
        for idx in sorted(label_counts):
            print(f" - {idx2lbl[idx]}: {label_counts[idx]} exemplo(s)")

        use_amp = torch.cuda.is_available()

        for ep in range(1, epochs + 1):
            self.train()
            loss_sum = corr = total = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                if use_amp:
                    with autocast():
                        out = self(x)
                        loss = criterion(out, y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    out = self(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()

                loss_sum += loss.item()
                preds = out.argmax(1)
                corr += (preds == y).sum().item()
                total += y.size(0)

            print(f"📘 Epoch {ep:2d}/{epochs} | Loss: {loss_sum/len(train_loader):.4f} | Acc: {100*corr/total:.2f}%")

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
        plt.savefig(f"confusion_matrix_inception_{name.lower()}.png")
        plt.close()

        print("\n".join(report))
        with open("relatorio_resultado_inception.txt", "a") as f:
            f.write("\n".join(report) + "\n\n")

        return acc

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"✅ Modelo salvo em: {path}")
