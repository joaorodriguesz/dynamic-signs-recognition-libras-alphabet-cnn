import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ResnetCustom(nn.Module):
    def __init__(self, num_classes: int, fine_tune=True):
        super().__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Ajusta primeira camada para entrada com 1 canal
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Libera ou congela fine-tuning
        for param in base_model.parameters():
            param.requires_grad = fine_tune

        self.backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool  
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),               
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.scaler = GradScaler()

    def forward(self, x):
        B, T, H, W = x.shape
        x = x.view(B * T, 1, H, W)  
        x = self.backbone(x)       
        x = self.classifier(x)      
        x = x.view(B, T, -1).mean(dim=1)  
        return x

    def train_model(self, train_loader, val_loader, idx2lbl, device, *, epochs=15, lr=1e-4):
        self.to(device)
        
        label_counts = Counter(lbl for _, lbl in train_loader.dataset.data)
        total = sum(label_counts.values())
        weights = torch.tensor(
            [total / label_counts[i] for i in range(len(idx2lbl))],
            dtype=torch.float32, device=device
        )
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

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
        plt.savefig(f"confusion_matrix_restnet_{name.lower()}.png")
        plt.close()

        print("\n".join(report))
        with open("relatorio_resultado_resnet.txt", "a") as f:
            f.write("\n".join(report) + "\n\n")

        return acc

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"✅ Modelo salvo em: {path}")
