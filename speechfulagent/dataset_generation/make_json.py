import json
from typing import List
from dataclasses import asdict

from speechfulagent.dataclasses import Experience


def make_json(dataset: List[List[Experience]], tails_length: List[int]) -> str:
    data = [
        {
            "id": i,
            "sequence": [exp.dict() for exp in seq],
            "tail": tail,
            "explanation": []
        }
        for i, (seq, tail) in enumerate(zip(dataset, tails_length))
    ]
    return json.dumps(data)