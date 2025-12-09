
# --- ЧАСТЬ 0: ПОДГОТОВКА ---

# Установка необходимых библиотек
!pip install biopython transformers torch

# Импорты
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import PDBParser, Selection, PDBIO # Та самая Biopython про которую я говорил. Ваша задача - разобраться с этой библиотекой в первую очередь
from Bio.SeqUtils import seq1
from transformers import AutoTokenizer, EsmModel
from Bio.PDB import PDBParser, Selection, PDBIO, NeighborSearch

# Настройки для графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

print("Все библиотеки готовы к работе!")

# --- ЧАСТЬ 1: СТРУКТУРНЫЙ АНАЛИЗ (НАЧАЛО) ---

# Выберите ваш PDB ID и ID лиганда
PDB_ID = "3MG1"
LIGAND_ID = "ECH" # Beta-carotene

# Загружаем структуру из PDB
!wget -q https://files.rcsb.org/download/{PDB_ID}.pdb

# Создаем парсер и загружаем структуру
parser = PDBParser(QUIET=True)
structure = parser.get_structure(PDB_ID, f"{PDB_ID}.pdb")
model = structure[0] # Берем первую модель из PDB

# --- ВАШ КОД ЗДЕСЬ ---
ligand = None
# Перебираем все остатки в структуре
for chain in model:
    for residue in chain:
        # Проверяем по названию остатка
        if residue.get_resname() == LIGAND_ID:
            ligand = residue
            break
    if ligand:
        break

if ligand:
    ligand_atoms = list(ligand.get_atoms())
    print(f"Лиганд {LIGAND_ID} найден в цепи {ligand.get_parent().id}.")
else:
    print(f"Ошибка: Лиганд {LIGAND_ID} не найден.")

# 2. Найдите все атомы белка.
#    Подсказка: можно использовать Selection.unfold_entities(model, 'A') для получения списка атомов
protein_atoms = []
# ... ваш код ...
protein_atoms = Selection.unfold_entities(model, 'A') # Получаем список всех атомов
print(f"Найдено {len(protein_atoms)} атомов белка.")

# 3. Напишите функцию, которая находит остатки кармана
def find_pocket_residues(protein_atoms, ligand, cutoff=4.5):
    """
    Находит все аминокислотные остатки, у которых есть хотя бы один атом
    на расстоянии <= cutoff от любого атома лиганда.
    Возвращает список объектов Residue.
    """
    pocket_residues = set() # Используем set, чтобы избежать дубликатов (это как массив, но элементы в нем не могут повторяться)

    if not ligand_atoms:
        return []

    # 1. Извлекаем все атомы лиганда
    ligand_atom_list = ligand_atoms

    # 2. Создаем структуру для поиска соседей (ускоряет процесс)
    # Используем NeighborSearch для быстрого нахождения атомов белка рядом с лигандом
    # Сначала извлечем координаты всех атомов лиганда для создания объекта.

    # 3. Находим все атомы белка, близкие к лиганду
    ns = NeighborSearch(protein_atoms)

    # Ищем атомы белка в пределах cutoff от любого атома лиганда
    # Эта функция делает за нас вложенный цикл:
    # Ищет ближайшие атомы белка для каждого атома лиганда.
    close_protein_atoms = set()
    for l_atom in ligand_atom_list:
        neighbors = ns.search(l_atom.get_coord(), cutoff, level='A') # level='A' - возвращает список атомов
        close_protein_atoms.update(neighbors)

    # 4. Извлекаем уникальные родительские остатки (Residue)
    pocket_residues = set()
    for atom in close_protein_atoms:
        pocket_residues.add(atom.get_parent())

    return list(pocket_residues)

# Получаем остатки кармана
pocket = find_pocket_residues(protein_atoms, ligand)
print(f"Найдено {len(pocket)} остатков в кармане.")

# 4. Сделайте скриншот из PyMOL и вставьте его в текстовую ячейку ниже.
