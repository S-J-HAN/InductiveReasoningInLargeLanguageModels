from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple

import openai
import cohere
import os
import requests
import time
import config

import numpy as np


class LLMReasoner(ABC):
    """Query LLM APIs"""

    def __init__(self, model: str):
        self.model = model

    @property
    def vendor(self) -> str:
        return ""
    
    @property
    def api_type(self) -> str:
        return ""
    
    @property
    def name(self) -> str:
        return f"{self.model}-{self.vendor}-{self.api_type}"
    
    @abstractmethod
    def generate_response(
        self,
        prompt: Any,
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> str:
        """Pass a prompt to LLM API and return its response as a string"""
        pass

    def generate_rating(
        self,
        prompt: Any,
        temperature: int = config.DEFAULT_TEMPERATURE,
        num_tries: int = 5,
        sleep_time: float = 10,
    ) -> Tuple[Any, Optional[float]]:
        """Make multiple attempts at getting a rating from an LLM API"""
        for _ in range(num_tries):
            try:
                raw_completion, rating = self._generate_rating(prompt, temperature)
                time.sleep(config.SLEEP_TIMES[self.vendor])
                return raw_completion, rating
            except:
                time.sleep(sleep_time)
        return None
    
    @abstractmethod
    def _generate_rating(
        self,
        prompt: Any,
        temperature: int,
    ) -> Tuple[Any, Optional[float]]:
        """Pass a prompt that asks for a rating to LLM API and return its raw response and parsed/processed rating"""
        pass

    @classmethod
    def calculate_completion_rating(cls, completion_logprobs: Dict[str, float]) -> float:
        """Given a dictionary of completion probabilities for numerical ratings at a single token position, 
        converts this into a single rating by taking a probability-weighted average."""
        
        completion_probs = {k: np.exp(v) for k,v in completion_logprobs.items()}
        s = sum(completion_probs.values())
        completion_probs = {k: v/s for k,v in completion_probs.items()}

        rating = 0
        for k,v in completion_probs.items():
            try:
                rating += float(k)*v
            except:
                pass
        
        return rating

    @classmethod
    def parse_chat_rating(cls, response: str) -> float:
        """
        Given a discrete response to a prompt that asks for a numerical rating, find the rating within the response and return as a float.
        Ideally the entire response is just the rating. If not, look for string matches to potential ratings.
        If there are multiple parsed ratings and one of them is 100, disregard the 100 (example: 'this argument's strength is 1 out of 100', the rating is '1' not '100').
        If there are still multiple parsed ratings, take the average (this never actually happens)
        """

        split_tokens = response.split(" ")
        split_tokens = [st.strip(",").strip(".") for st in split_tokens]
        floats = []
        for st in split_tokens:
            try:
                floats.append(float(st))
            except:
                pass
        if len(floats) == 1:
            return floats[0]
        floats = [f for f in floats if f != 100]
        if len(floats) == 1:
            return floats[0]
        else:
            return np.mean(floats)


class OpenAIChatReasoner(LLMReasoner):
    """'Chat' style OpenAI models like GPT-4 and GPT-3.5"""

    @property
    def vendor(self) -> str:
        return "openai"
    
    @property
    def api_type(self) -> str:
        return "chat"
    
    def generate_response(
        self,
        prompt: Any,
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> str:
        
        openai.api_key = os.environ['OPENAI']
        completion = openai.ChatCompletion.create(
              model=self.model,
              messages=prompt,
              temperature=temperature,
            )

        return completion.choices[0].message.content

    def _generate_e2_prompt_rating(
        self, 
        prompt: List[Dict[str, str]], 
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> Tuple[Any, Optional[float]]:

        openai.api_key = os.environ['OPENAI']
        assistant_message = self.generate_response(prompt, temperature)

        return assistant_message, self.parse_chat_rating(assistant_message)
    

class OpenAICompletionReasoner(LLMReasoner):
    """'Completion' style OpenAI models like text-danvinci-003"""

    @property
    def vendor(self) -> str:
        return "openai"
    
    @property
    def api_type(self) -> str:
        return "completion"
    
    def generate_response(
        self,
        prompt: Any,
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> str:
        
        openai.api_key = os.environ['OPENAI']
        completion = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=100,
            logprobs=5,
        )

        return completion["choices"][0]["text"]

    def _generate_e2_prompt_rating(
        self, 
        prompt: str, 
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> Tuple[Any, Optional[float]]:

        openai.api_key = os.environ['OPENAI']
        completion = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=100,
            logprobs=5,
        )

        return completion, self.calculate_completion_rating(completion["choices"][0].logprobs.top_logprobs[-1])


class CohereCompletionReasoner(LLMReasoner):

    @property
    def vendor(self) -> str:
        return "cohere"
    
    @property
    def api_type(self) -> str:
        return "completion"
    
    def generate_response(
        self,
        prompt: Any,
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> str:
        co = cohere.Client(os.environ['COHERE'])
        return co.generate(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=100,
        ).generations[0].text

    def _generate_e2_prompt_rating(
        self, 
        prompt: str, 
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> Tuple[Any, Optional[float]]:

        prediction = self.generate_response(prompt, temperature)

        return prediction, self.parse_chat_rating(prediction)
    

class TextSynthCompletionReasoner(LLMReasoner):

    @property
    def vendor(self) -> str:
        return "textsynth"

    @property
    def api_type(self) -> str:
        return "completion"
    
    def generate_response(
        self,
        prompt: Any,
        temperature: int = 0.11,
    ) -> str:
        api_url = "https://api.textsynth.com"
        response = requests.post(api_url + "/v1/engines/" + self.model + "/completions", headers = { "Authorization": "Bearer " + os.environ['TEXTSYNTH'] }, json = { "prompt": prompt, "max_tokens": 100, "temperature": temperature })
        resp = response.json()
        if "text" in resp: 
            return resp["text"]
        else:
            return None

    def _generate_e2_prompt_rating(
        self, 
        prompt: str, 
        temperature: int = 0.11,
    ) -> Tuple[Any, Optional[float]]:
        
        completion = self.generate_response(prompt, temperature)

        return completion, self.parse_chat_rating(completion)
