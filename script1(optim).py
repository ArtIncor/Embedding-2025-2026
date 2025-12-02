import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, EsmModel
from Bio import Entrez, SeqIO
from scipy.spatial.distance import cosine



#-----ЧАСТЬ 0: ЗАГРУЗКА БЕЛКА ИЗ БАЗЫ NCBI ПО ID-----




# Email для запросов к NCBI (требование API)
Entrez.email = "artemiy.kosinets2007@gmail.com"

# Используем GPU NVIDIA, если есть, иначе CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Вычисления выполняются на: {device}")
if device.type == 'cuda':
    print(f"Видеокарта: {torch.cuda.get_device_name(0)}")




# ----- ЧАСТЬ 1: АВТОМАТИЧЕСКОЕ ПОЛУЧЕНИЕ ПОСЛЕДОВАТЕЛЬНОСТИ -----




accession_id = "BAN66288.1"
protein_sequence = ""  # Инициализация переменной для безопасности
protein_name = "Unknown Protein"

try:
    handle = Entrez.efetch(db="protein", id=accession_id, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    handle.close()

    protein_sequence = str(record.seq)
    # Более надежная обработка имени
    description_parts = record.description.split(']')
    protein_name = description_parts[0] + ']' if len(description_parts) > 1 else record.description
    #protein_name = record.description.split(']')[0] + ']' ##Можно оставить как у Макара,
    ##но Gemini советует так более безопасно

    print(f"Успешно загружен белок: {protein_name}")
    print(f"Длина: {len(protein_sequence)} а.о.")
    print(f"Начало последовательности: {protein_sequence[:30]}...")
except Exception as e:
    print(f"Ошибка при загрузке: {e}")
    # Останавливаем код, если белок не загружен, чтобы не было ошибок
    exit()




# ----- ЧАСТЬ 2: СРАВНЕНИЕ "ПОРТРЕТОВ" ОТ РАЗНЫХ МОДЕЛЕЙ -----




model_names = [
    "facebook/esm2_t6_8M_UR50D",  # Маленькая (8 млн параметров)
    "facebook/esm2_t12_35M_UR50D",  # Средняя (35 млн)
    "facebook/esm2_t30_150M_UR50D"  # Большая (150 млн)
]

# Переменная для хранения эмбеддингов последней модели (для Части 3)
embeddings = 0

for name in model_names:
    print(f"\n--- Обработка моделью: {name} ---")

    # Загружаем токенайзер и модель
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = EsmModel.from_pretrained(name).to(device)  # Отправляем модель на GPU
    model.eval()  # Переводим модель в режим оценки (выключает dropout и т.д.)

    # Токенизация и отправка тензоров на GPU
    inputs = tokenizer(protein_sequence, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[0, 1:-1, :]

    # Переводим тензор в numpy массив для вычислений и графиков
    # Важно: .cpu() переносит данные с видеокарты в RAM
    data = embeddings.cpu().numpy()

    # Находим 2-й и 98-й перцентили
    vmin = np.percentile(data, 2)
    vmax = np.percentile(data, 98)

    plt.figure(figsize=(15, 6))
    # Добавляем аргументы vmin и vmax в функцию imshow
    plt.imshow(data.T, cmap='BuGn', aspect='auto', vmin=vmin, vmax=vmax)
    plt.title(f"Портрет '{protein_name}'\nМодель: {name}")
    plt.xlabel("Позиция аминокислоты")
    plt.ylabel("Измерение в эмбеддинге")
    plt.colorbar()

    print("Отображение графика... Закройте окно графика, чтобы продолжить.")
    plt.show()
    plt.close()  # Очищаем фигуру из памяти после закрытия окна




# ----- ЧАСТЬ 3: СХОДСТВО ФУНКЦИОНАЛЬНЫХ УЧАСТКОВ -----



print("\n--- Анализ функциональных участков ---")

# Используем эмбеддинги от самой большой модели для точности
# Примечание: embeddings сейчас - это PyTorch Tensor на GPU (так как мы не перезаписали его numpy версией)
embeddings_large_model = embeddings

# Координаты участков
site1_start, site1_end = 0, 19
site2_start, site2_end = 20, 222

# Вырезаем эмбеддинги для этих участков (слайсинг работает на GPU очень быстро)
site1_embeddings = embeddings_large_model[site1_start:site1_end + 1]
site2_embeddings = embeddings_large_model[site2_start:site2_end + 1]

# Усредняем векторы. Результат все еще на GPU.
site1_vector_gpu = site1_embeddings.mean(dim=0)
site2_vector_gpu = site2_embeddings.mean(dim=0)

# Для функции cosine из scipy нам нужны numpy массивы на CPU
site1_vector = site1_vector_gpu.cpu().numpy()
site2_vector = site2_vector_gpu.cpu().numpy()

# Считаем косинусное сходство
similarity = 1 - cosine(site1_vector, site2_vector)

print(
    f"Косинусное сходство между участком 1 ({site1_start}-{site1_end}) и участком 2 ({site2_start}-{site2_end}): {similarity:.4f}")