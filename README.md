# SpeechfulAgent
Выпускная квалификационная работа бакалавра, кафедра ИУ3

**Студент**: Сергей Козлов
**Период обучения**: 2022-2026 года

Алгоритм сильно основан на коде, предложенном в книге Maxim Lapan Deep Reinforcement Learning Hands-On

Версия Python 3.11.9

Использование

```
python demo.py --help

usage: demo.py [-h] [-g GAMMA] [-a ALPHA] [-s SAVE_DIR] [-v VERSION] [-o OBJECTIVE] [-t TEST_EPISODES] [--verbose] [--video VIDEO]
               {train,play}

positional arguments:
  {train,play}

options:
  -h, --help            show this help message and exit
  -g GAMMA, --gamma GAMMA
  -a ALPHA, --alpha ALPHA
  -s SAVE_DIR, --save-dir SAVE_DIR
  -v VERSION, --version VERSION
  -o OBJECTIVE, --objective OBJECTIVE
  -t TEST_EPISODES, --test-episodes TEST_EPISODES
  --verbose
  --video VIDEO
```
параметры обучения:

GAMMA - коэффициент гамма

ALPHA - коэффициент альфа

OBJECTIVE - цель reward, которую необходимо достичь

TEST_EPISODES - количество эпизод, проигрываемых для валидации

SAVE_DIR - каталог для сохранения моделей

VERBOSE - флаг для включения логгирования в консоль

параметры пробной игры:

SAVE_DIR - каталог для сохранения моделей

VERSION - версия (по умолчанию последняя)

VIDEO - каталог для сохранения видео

Пример запуска:

```
python demo.py train -g 0.99 -a 0.7 -s models --verbose
```
```
python demo.py play -s models --video videos
```