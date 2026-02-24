# Структура пакета
```
speechfulagent
│   dataclasses.py
│   README.md
│   speechfulagent.py
│   types.py
│   versioning.py
│   __init__.py
│
├───agent
│       actor.py
│       agent.py
│       critic.py
│       __init__.py
│
├───explainer
│   │   explainer.py
│   │   __init__.py
│   │
│   ├───preprocessing
│   │       sequence_embed.py
│   │       tokenizer.py
│   │       __init__.py
│   │
│   └───transformer
│           model.py
│           positional_encoder.py
│           state_encoder.py
│           __init__.py
│
└───train
    │   __init__.py
    │
    ├───agent
    │       trainer.py
    │       wrappers.py
    │       __init__.py
    │
    └───explainer
            dataset.py
            trainer.py
            __init__.py

```

dataclasses.py - полезные датаклассы решения.

speechfulagent.py - основной класс агента.

types.py - объявления типов.

versioning.py - класс-примесь для версионирования.

agent - пакет с модулями для построения агента.

actor.py - нейронная сеть актора.

agent.py - агент.

critic.py - нейронная сеть критика.

explainer - пакет с модулями для построения объяснителя.

preprocessing - подпакет с модулями предобработки данных.

sequence_embed.py - содержит функцию для получения представления последовательности действий агента.

tokenizer.py - содержит класс токенизатор для работы с тектовыми данными.

transformer - подпакет с модулями трансформера.

model.py - содержит саму модель трансформера.

positional_encoder.py - содержит класс для позиционного кодирования входной последовательности.

state_encoder.py - энкодер-блок трансформера для преобразования двух последовательностей.

train - пакет с модулями для организации процесса обучения.

agent - подпакет с модулями для обучения агента.

trainer.py - объявление класса AgentTrainer, который производит обучение агента.

wrappers.py - обёртки для среды.

explainer - подпакет с модулями для обучения объяснителя.

dataset.py - объяление класса Dataset из pytorch для работы с набором данных.

trainer.py - объявление класса ExplainerTrainer, который производит обучение объяснителя.