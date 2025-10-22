# Структура пакета
```
speechfulagent
│   dataclasses.py
│   play.py
│   README.md
│   types.py
│   versioning.py
│   __init__.py
│
├───agent
│       agent.py
│       net.py
│       __init__.py
│
└───train
        replay_buffer.py
        trainer.py
        wrappers.py
        __init__.py
```

dataclasses.py - полезные датаклассы решения.

play.py - функция для запуска пробной игры агента

types.py - объявление типов

versioning.py - класс-примесь для версионирования

agent - пакет с модулями для построения агента

agent.py - агент

net.py - нейронная сеть для агента

train - пакет с модулями для организации процесса обучения

replay_buffer.py - реплей буффер для сохранения sars' кортежей

trainer.py - объявление класса Trainer, который производит обучение агента

wrappers.py - обёртки для среды