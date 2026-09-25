#BBBB  IIIII N   N  AAA  RRRR  Y   Y    CCCC L      AAA  SSSSS SSSSS IIIII FFFFF IIIII EEEEE RRRR
#B  BB   I   NN  N A   A R  RR  Y Y    CC    L     A   A S     S       I   F       I   E     R  RR
#BBBB    I   N N N AAAAA RRRR    Y     C     L     AAAAA  SSSS  SSSS   I   FFF     I   EEE   RRRR
#B  BB   I   N  NN A   A R  R    Y     CC    L     A   A     S     S   I   F       I   E     R  R
#BBBB  IIIII N   N A   A R   R   Y      CCCC LLLLL A   A SSSSS SSSSS IIIII F     IIIII EEEEE R   R   #for protein in PSP


#💀 - alpha 1.0    - 01.04.26 - Неправильная структура датасетов в коде
#alpha 1.1    - 01.04.26 - убрали PyTorch Lightning тк сложная логика
#alpha 1.2    - 04.07.26 - Есть недочеты по колву классов + $logits = model(x).squeeze()$ теперь с -1 - Dima
#alpha 1.3    - 12.07.26
#Authors: Kosinets A.D. (+ Dmitry S. error protect)




import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

# пути к файлам
POSITIVE = "/home/jupyter-mulberry/positive_Artema.h5"
NEGATIVE = "/home/jupyter-mulberry/negative_Artema.h5"


BATCH_SIZE = 256
NUM_WORKERS = 8
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3
DROPOUT = 0.3
NUM_FOLDS = 5
RANDOM_SEED = 67 #SIIIIIIIX SEEEEVEEEEEN
OUTPUT_DIR = Path("results_Artem")
OUTPUT_DIR.mkdir(exist_ok=True)



torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)




class H5Dataset(Dataset):
    def __init__(self, positive_file, negative_file):
        self.pos_h5 = h5py.File(positive_file, 'r')
        self.neg_h5 = h5py.File(negative_file, 'r')
        #----------------
        self.samples = []
        for pdb_id in self.pos_h5.keys():
            for chain in self.pos_h5[pdb_id].keys():
                self.samples.append((self.pos_h5, pdb_id, chain, 1))
        for pdb_id in self.neg_h5.keys():
            for chain in self.neg_h5[pdb_id].keys():
                self.samples.append((self.neg_h5, pdb_id, chain, 0))   
        print(f"Датасет загружен: {len(self.samples)} цепей всего  "
              f"({len([s for s in self.samples if s[3]==1])} позитивов,  "
              f"{len([s for s in self.samples if s[3]==0])} негативов) ")
    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        h5_file, pdb_id, chain, label = self.samples[idx]
        try:
            emb = h5_file[pdb_id][chain]['emb'][:]
            emb_tensor = torch.from_numpy(emb.astype(np.float32))
            emb_mean = emb_tensor.mean(dim=0)   # Mean Pooling
        except Exception as e:
            print(f"Ошибка при чтении {pdb_id}/{chain}: {e} ")
            emb_mean = torch.zeros(1280, dtype=torch.float32)
        return emb_mean, torch.tensor(label, dtype=torch.float32)
    def close(self):
        self.pos_h5.close()
        self.neg_h5.close()



#------------------------------------
class MyNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.model(x)
#----------------------------------
dataset = H5Dataset(POSITIVE, NEGATIVE)
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
labels = np.array([sample[3] for sample in dataset.samples])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n{'='*20} FOLD {fold+1}/{NUM_FOLDS} {'='*20} ")
 #----------   
    # баланс классов далее ->
    #-------------------------------
    train_labels = np.array([labels[i] for i in train_idx])
    class_counts = np.bincount(train_labels.astype(int))
    weights = 1. / class_counts
    sample_weights = weights[train_labels.astype(int)]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    #-------------------------------



    model = MyNet(dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    best_f1 = 0.0
    best_model_path = None
    patience_counter = 0
    


    #-------=---логи чеки ->
    fold_dir = OUTPUT_DIR / f"fold_{fold}"
    fold_dir.mkdir(exist_ok=True)
    for epoch in range(MAX_EPOCHS):
        # ---TRAIN -------------------------
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            

            train_loss += loss.item()
            

        # --- VAL -----------------------------
        model.eval()
        val_loss = 0.0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze(-1)
                loss = criterion(logits, y)
                val_loss += loss.item()
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())
                

        #----------
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels).numpy()
        probs = torch.sigmoid(all_logits).numpy()
        preds = (probs > 0.5).astype(int)
        



        val_acc = accuracy_score(all_labels, preds)
        val_f1 = f1_score(all_labels, preds)
        val_prec = precision_score(all_labels, preds, zero_division=0)
        val_rec = recall_score(all_labels, preds, zero_division=0)
        


        try:
            val_auroc = roc_auc_score(all_labels, probs)
        except ValueError:
            val_auroc = 0.0 # если в валидации только один класс - Dima
        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Acc: {val_acc:.4f} | F1: {val_f1:.4f} | "
              f"Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | AUROC: {val_auroc:.4f}")
              
        #---------------------------------------
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_path = fold_dir / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1    
        if patience_counter >= 8:
            print("Early stopping!")
            break



    #----------------------------
    best_nigga = best_model_path
    print(f"Лучшая модель фолда {fold+1} сохранена: {best_nigga} ")
#-----------------------
dataset.close()
print("\n5-fold Cross-Validation завершена! ")
print(f"Результаты сохранены в папке: {OUTPUT_DIR} ")