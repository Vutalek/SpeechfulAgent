# SpeechfulAgent
Выпускная квалификационная работа бакалавра, кафедра ИУ3

**Студент**: Сергей Козлов
**Период обучения**: 2022-2026 года

Алгоритм сильно основан на коде, предложенном в книге Maxim Lapan Deep Reinforcement Learning Hands-On

Версия Python 3.11.9

Использование

```
python train.py --help
usage: train.py [-h] [--model-dir MODEL_DIR] [--verbose]

options:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
  --verbose
```

```
python demo.py --help 
usage: demo.py [-h] [--model-dir MODEL_DIR] [--version VERSION] [--video VIDEO]

options:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
  --version VERSION
  --video VIDEO
```

Релизован алгоритм Deep Q-Learning с модификациями:
- использован Replay Buffer для обучения на последних данных;
- использована техника epsilon decay для того, чтобы обеспечить агента случайными действиями в самом начале;
- использована техника с двумя моделями: обучаемая и целевая модель. Целевая модель служит для оценки значения следующих состояний.

Также реализовано версионирование моделей (папка models текущего репозитория) с автоинкрементом номера модели. В каждой версии содержится два файла: 
- weights.dat - веса модели
- info.yml - информация по процессу обучения (количество итераций, гиперпараметры)

параметры обучения:

MODEL_DIR - каталог для сохранения моделей

VERBOSE - флаг для включения логгирования в консоль

параметры пробной игры:

MODEL_DIR - каталог из которой читаются модели

VERSION - версия (по умолчанию последняя)

VIDEO - каталог для сохранения видео

Пример запуска:

```
python train.py --model-dir models --verbose
```
```
python demo.py --model-dir models --video videos
```