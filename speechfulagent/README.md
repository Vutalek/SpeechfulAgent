# Пакет `speechfulagent`

Пакет содержит код для обучения агентов с подкреплением и генерации объяснений их поведения. Центральные сущности проекта - RL-агент, эпизод взаимодействия со средой и объяснитель, который превращает эпизод в текст.

## Структура пакета

```text
speechfulagent/
|-- agent/
|   |-- base_agent.py
|   |-- dqn.py
|   |-- a2c.py
|   |-- ddpg.py
|   |-- ppo.py
|   `-- net/
|       |-- dqn.py
|       |-- a2c/
|       |-- ddpg/
|       `-- ppo/
|-- explainer/
|   |-- base_explainer.py
|   |-- openai_explainer.py
|   |-- r3l/
|   |   |-- r3l_explainer.py
|   |   `-- r3l_state_encoder.py
|   `-- tokenizer/
|-- train/
|   |-- agent/
|   |   |-- dqn.py
|   |   |-- a3c.py
|   |   |-- ddpg.py
|   |   |-- ppo.py
|   |   `-- replay_buffer.py
|   `-- explainer/
|       |-- dataset.py
|       |-- r3l_trainer.py
|       `-- trainer.py
|-- dataclasses.py
|-- speechfulagent.py
|-- types.py
`-- versioning.py
```

## Корневые файлы

- `dataclasses.py` - общие структуры данных: `Experience`, `EnvInfo` и dataclass-объекты с параметрами обучения.
- `types.py` - алиасы типов `State` и `Action`.
- `versioning.py` - миксин для сохранения и загрузки моделей по версиям `v1`, `v2`, ...
- `speechfulagent.py` - фасад `SpeechfulAgent`, который запускает агента, накапливает эпизод и периодически вызывает объяснитель.
- `__init__.py` - инициализатор пакета; сейчас не содержит явных экспортов.

## `agent`

`BaseAgent` хранит ссылку на Gymnasium environment, определяет типы пространств наблюдений/действий, управляет seed, состоянием среды и суммарной наградой.

Реализации:

- `DQNAgent` - выбирает действие по Q-values или epsilon-greedy exploration.
- `A2CAgent` - использует actor-critic сеть, поддерживает дискретные и непрерывные действия.
- `DDPGAgent` - использует actor/critic и OU-процесс для exploration в непрерывных средах.
- `PPOAgent` - генерирует действия из policy-сети для дискретного или непрерывного action space.

`agent/net` содержит PyTorch-модели, которые используются агентами и тренерами: DQN MLP, A2C-сети, DDPG actor/critic и PPO actor/critic.

## `train.agent`

Тренеры реализуют полный цикл обучения:

- `DQNTrainer` - replay buffer, target network, epsilon decay.
- `A3CTrainer` - несколько worker-процессов, накопление градиентов и обновление общей A2C-сети.
- `DDPGTrainer` - actor/critic loss, target networks и soft update.
- `PPOTrainer` - сбор траекторий, generalized advantage estimation и clipped PPO objective.
- `ReplayBuffer` - хранение и случайная выборка опыта.

Все тренеры возвращают кортеж: обученный агент, `EnvInfo` и dataclass с параметрами обучения.

## `explainer`

`BaseExplainer` задает единый метод `generate(prompt, context, max_tokens, temperature, top_k, ...)`.

Реализации:

- `OpenaiExplainer` сериализует эпизод в JSON и отправляет его в OpenAI-compatible Chat Completions API.
- `R3LExplainer` кодирует траекторию в embedding-последовательность и продолжает ее causal language model.
- `R3LStateEncoder` преобразует последовательности состояний, действий и наград через линейные блоки, LSTM, LayerNorm и проекцию в размерность LLM.

## `train.explainer`

- `ExperienceDataset` читает JSON-эпизоды и пары "эпизод - объяснение" из `final_dataset`.
- `R3LExplainerTrainer` обучает state encoder для R3L-подхода. LLM загружается через `transformers`, а loss считается по целевому тексту объяснения.
- `trainer.py` - ранний экспериментальный trainer для transformer-based объяснителя.

## Поток данных R3L

```text
JSON episode + explanation
        |
        v
ExperienceDataset
        |
        v
collation_fn: states/actions/rewards -> tensors
        |
        v
R3LStateEncoder
        |
        v
LLM embeddings + teacher forcing
        |
        v
loss по токенам объяснения
```

Такой подход позволяет обучать небольшой encoder, который переводит траекторию агента в пространство языковой модели, не дообучая всю LLM.
