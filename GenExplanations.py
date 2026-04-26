import os
import json
from typing import List, Dict

from openai import OpenAI
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

ya_api_key = os.environ.get("YA_API_KEY")
client = OpenAI(
    api_key=ya_api_key,
    base_url="https://llm.api.cloud.yandex.net/v1"
)

prompt = """Ты RL-исследователь.
    Тебе будет предоставлен эпизод игры в среде FrozenLake.
    FrozenLake - это среда, в которой эльф должен добраться до подарка по скользкому озеру.
    При совершении шага в среде, он имеет вероятность 0.33 поскользнуться направо от выбранного направления, вероятность 0.33 поскользнуться налево от выбранного направления и вероятность 0.33 пойти по выбранному пути.
    Он совершает действия, пронумерованные от 0 до 3:
    0 - двигаться налево
    1 - двигаться вниз
    2 - двигаться вправо
    3 - двигаться вверх
    Поле состоит из 16 клеток, составленных в квадрат 4 на 4 клетки.
    Клетки пронумерованы от 0 до 15, где 0 - это левый верхний угол, а 15 - это правый нижний угол.
    Эльф начинает с клетки 0. А его цель - достичь клетки 15.
    На поле расположены проруби, при попадании в прорубь, Эльф проигрывает. Проруби находятся в клетках 5, 7, 11, 12.
    Если Эльф достигает цели, то он получает награду 1.0, иначе он ничего не получает.

    Тебе будет предоставлена последовательность действий агента RL, который обучен проходить данную игру.
    Твоя задача - объяснить её по плану максимально кратко:
    - что мы наблюдаем в среде
    - что делает агент
    - почему он это делает - самый важный пункт
    Старайся избегать использование номеров клеток и действий. Старайся понять стратегию, которой руководствуется агент.

    Информация об эпизоде будет представлена в виде JSON-массива, состоящего из объектов с полями:
    - state - номер текущей клетки
    - action - принятое действие
    - reward - полученная награда от этого действия
    - done - флаг завершения
    """

def explain_folder(folder: str) -> List[Dict[str, str]]:
    explanations = []
    for filename in tqdm(os.listdir(folder)):
        with open(folder+"/"+filename, "rt") as f:
            episode = f.read()
        completion = client.chat.completions.create(
            model="gpt://b1gccpjnou3q4l9pegs9/yandexgpt-5.1/latest",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": episode}
            ]
        )
        explanations.append(
            {
                "folder": folder,
                "filename": filename,
                "explanation": completion.choices[0].message.content
            }
        )
    return explanations

if __name__ == "__main__":
    explanations_good = explain_folder("final_dataset/episodes_good/data")
    explanations_bad = explain_folder("final_dataset/episodes_bad/data")
    with open("final_dataset/explanations_good.json", "wt", encoding="utf-8") as f:
        json.dump(explanations_good, f, ensure_ascii=False, indent=4)
    with open("final_dataset/explanations_bad.json", "wt", encoding="utf-8") as f:
        json.dump(explanations_bad, f, ensure_ascii=False, indent=4)





