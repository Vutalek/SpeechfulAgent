import re
import json
import argparse
from typing import Tuple, Any, List

import matplotlib.pyplot as plt


def read_histories(base_dir: str) -> Tuple[Any, Any]:
    train = base_dir + "/loss_history.json"
    val = base_dir + "/validation_history.json"
    with open(train, "rt") as f:
        train_history = json.load(f)
    with open(val, "rt") as f:
        val_history = json.load(f)
    return train_history, val_history

def get_epochs(series) -> List[float]:
    result = []
    for epoch in series:
        result.append(
            sum(epoch) / len(epoch)
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="version directory")
    
    args = parser.parse_args()
    train, val = read_histories(args.dir)

    train_epoch = get_epochs(train)
    val_epoch = get_epochs(val)

    # Совместный график
    x = range(len(train_epoch))
    fig = plt.figure(figsize=(12, 5))
    plt.plot(x, train_epoch, label="Обучение", color="purple")
    plt.plot(x, val_epoch, label="Валидация", color="green")
    plt.xlabel("Эпоха")
    plt.ylabel("Средняя потеря, нат")
    plt.legend()
    plt.grid()
    plt.savefig("train_val.png")

    # Валидация по итерациям
    val_iter = []
    for v in val:
        val_iter.extend(v)
    fig = plt.figure(figsize=(12, 5))
    x = range(len(val_iter))
    plt.plot(x, val_iter, color="purple")
    plt.xlabel("Итерация")
    plt.ylabel("Потеря, нат")
    plt.grid()
    plt.savefig("val_iter.png")

    # Валидация по эпохам
    train_iter = []
    for t in train:
        train_iter.extend(t)
    fig = plt.figure(figsize=(12, 5))
    x = range(len(train_iter))
    plt.plot(x, train_iter, color="purple")
    plt.xlabel("Итерация")
    plt.ylabel("Потеря, нат")
    plt.grid()
    plt.savefig("train_iter.png")