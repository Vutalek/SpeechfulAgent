# SpeechfulAgent

SpeechfulAgent - выпускной проект, посвященный обучению агентов с подкреплением и генерации текстовых объяснений их поведения. Репозиторий содержит реализации RL-агентов, тренеров, нейросетевых моделей, набор эпизодов FrozenLake и код объяснителя, который преобразует траекторию агента в человекочитаемое описание.

**Студент:** Сергей Козлов  
**Период обучения:** 2022-2026  
**Рекомендуемая версия Python:** 3.11.9

Часть реализаций алгоритмов обучения с подкреплением опирается на подходы из книги Maxim Lapan, *Deep Reinforcement Learning Hands-On*.

## Что делает проект

Проект решает две связанные задачи:

1. Обучает RL-агентов для сред Gymnasium с дискретными и непрерывными пространствами состояний/действий.
2. Строит объяснитель, который получает последовательность опытов агента (`state`, `action`, `reward`, `next_state`, `done`) и генерирует текстовое объяснение стратегии поведения.

Основной сценарий выглядит так:

```text
Gymnasium environment
        |
        v
RL agent makes steps
        |
        v
episode: List[Experience]
        |
        v
explainer generates text
```

## Структура проекта

```text
.
|-- agent_models/              # сохраненные версии обученных агентов
|-- explainer_models/          # сохраненные версии объяснителей и история обучения
|-- final_dataset/             # эпизоды и текстовые объяснения для обучения explainer
|-- runs/                      # логи TensorBoard
|-- scripts/
|   `-- train_explainer.py     # пример запуска обучения R3L-объяснителя
|-- speechfulagent/
|   |-- agent/                 # агенты DQN, A2C, DDPG, PPO
|   |-- explainer/             # базовый интерфейс объяснителя, OpenAI и R3L explainers
|   |-- train/                 # тренеры агентов и объяснителей
|   |-- dataclasses.py         # структуры данных проекта
|   |-- speechfulagent.py      # фасад для запуска агента с объяснителем
|   |-- types.py               # базовые типы State и Action
|   `-- versioning.py          # сохранение и загрузка версий моделей
|-- demo.py                    # демонстрационный запуск агента в Gymnasium
|-- requirements.txt           # зависимости
`-- README.md
```

## Основные модули

### `speechfulagent.dataclasses`

Содержит датаклассы, которыми обмениваются остальные части проекта:

- `Experience` - один шаг агента в среде: состояние, действие, награда, следующее состояние и флаг завершения.
- `EnvInfo` - краткое описание среды.
- `DQNTrainInfo`, `A3CTrainInfo`, `DDPGTrainInfo`, `PPOTrainInfo`, `R3LTrainInfo` - параметры и результаты обучения.

### `speechfulagent.agent`

Пакет с агентами:

- `DQNAgent` - агент для сред с дискретным пространством действий.
- `A2CAgent` - actor-critic агент для дискретных и непрерывных действий.
- `DDPGAgent` - агент для непрерывного пространства действий с OU-шумом для exploration.
- `PPOAgent` - агент PPO для дискретных и непрерывных действий.

Нейросетевые архитектуры лежат в `speechfulagent/agent/net`: MLP для DQN, actor/critic для DDPG и PPO, а также сети A2C.

### `speechfulagent.train.agent`

Пакет с процедурами обучения:

- `DQNTrainer` - обучение DQN с replay buffer и target network.
- `A3CTrainer` - асинхронное обучение A2C-сети несколькими worker-процессами.
- `DDPGTrainer` - actor-critic обучение для непрерывных действий с target networks.
- `PPOTrainer` - обучение PPO с GAE, clipping objective и несколькими эпохами по траектории.
- `ReplayBuffer` - буфер опыта для off-policy алгоритмов.

Тренеры возвращают обученного агента, описание среды и объект с параметрами обучения. Эти данные используются при сохранении версии модели.

### `speechfulagent.explainer`

Пакет объяснителей:

- `BaseExplainer` - общий интерфейс `generate(...)`.
- `OpenaiExplainer` - объяснитель через OpenAI-compatible Chat Completions API.
- `R3LExplainer` - локальный объяснитель, который кодирует траекторию агента и подает ее в causal language model.
- `R3LStateEncoder` - state/action/reward encoder: отдельные линейные модули, LSTM, LayerNorm и проекция в размерность embedding-пространства LLM.
- `SimpleTokenizer` и `BaseTokenizer` - простая инфраструктура токенизации для старых/экспериментальных объяснителей.

### `speechfulagent.train.explainer`

Пакет обучения объяснителей:

- `ExperienceDataset` читает `final_dataset/explanations_good.json` и `final_dataset/explanations_bad.json`, загружает соответствующие эпизоды и токенизирует целевые объяснения.
- `R3LExplainerTrainer` обучает только encoder-часть R3L-объяснителя, используя LLM как замороженную языковую модель.
- `trainer.py` содержит более раннюю экспериментальную версию обучения transformer-based объяснителя.

## Данные и модели

`final_dataset` содержит две группы эпизодов:

- `episodes_good` - успешные или качественные траектории агента;
- `episodes_bad` - неудачные или менее качественные траектории.

Файлы `explanations_good.json` и `explanations_bad.json` связывают эпизоды с текстовыми объяснениями. Эти пары используются для обучения R3L-объяснителя.

`agent_models` и `explainer_models` хранят версии сохраненных моделей. Основной механизм версионирования использует подпапки вида `v1`, `v2`, `v3` и файл `info.yml` с метаданными обучения; часть старых сохранений может отличаться по набору файлов.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Для обучения R3L-объяснителя по умолчанию используется модель `Qwen/Qwen3-1.7B` из Hugging Face. При первом запуске она должна быть доступна локально или скачана через `transformers`.

## Пример обучения объяснителя

```bash
python scripts/train_explainer.py
```

Скрипт создает `R3LExplainerTrainer`, читает `final_dataset`, обучает encoder один epoch и сохраняет новую версию в `explainer_models`.

## Пример запуска агента

`demo.py` показывает идею демонстрационного запуска агента в Gymnasium с записью видео:

```bash
python demo.py --agent-dir agent_models --agent-version latest --explainer-dir explainer_models --explainer-version latest
```

Перед запуском убедитесь, что выбранная среда совпадает с сохраненной моделью агента, а интерфейс загрузки моделей соответствует текущей версии кода.