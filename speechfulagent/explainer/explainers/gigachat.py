from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_gigachat import GigaChat
from torch import Tensor

from speechfulagent.dataclasses import *
from speechfulagent.explainer import BaseExplainer


class GigaChatExplainer(BaseExplainer):
    def __init__(
        self,
        prompt: PromptTemplate,
        credentials: str,
        model: str="GigaChat-Pro"
    ):
        super().__init__()
        self.model = model
        self.credentials = credentials
        self.prompt = prompt

    def generate(
        self, 
        prompt: Tensor, 
        context: Tensor, 
        max_length: int=32, 
        temperature: float=0, 
        top_k: int=0
    ) -> str:
        chat = GigaChat(
            max_tokens=max_length,
            temperature=temperature,
            credentials=self.credentials,
            verify_ssl_certs=False,
            model=self.model
        )