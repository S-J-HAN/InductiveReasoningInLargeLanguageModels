from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

import openai
import cohere
import os
import requests
import config

import numpy as np


class LLMReasoner(ABC):
    """Query LLM APIs"""

    def __init__(self, model: str):
        self.model = model
    
    @property
    def name(self) -> str:
        return f"{self.model}-completion"
    
    @abstractmethod
    def rate_e2_prompt(
        self,
        prompt: Any,
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> Optional[float]:
        """Pass inductive reasoning prompt to LLM API and return its rating"""
        pass

    def calculate_e2_rating(self, completion_logprobs: Dict[str, float]) -> float:
        """Given a dictionary of completion probabilities at a single token position, 
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

    def parse_e2_response(self, response: str) -> Optional[float]:
        """Given a discrete response to an experiment 2 question, find the actual rating and return as a float."""

        parsed_answer = None

        try:
            parsed_answer = float(response)
        except:
            prev_parses = []
            for i in range(100,-1,-1):
                for parse in [
                    f"Answer: {i}",
                    f"Likelihood: {i}",
                ]:
                    if parse in response:
                        parsed_answer = i
                        prev_parses.append((parse, parsed_answer))
                        break
                if parsed_answer:
                    break

            if not parsed_answer:
                for i in range(100,-1,-1):
                    for parse in [
                        f" {i}% likelihood",
                        f"around {i}.",
                        f"around {i},",
                        f"around {i} ",
                        f"likelihood of {i}.",
                        f"likelihood of {i} ",
                        f"value of {i}.",
                        f"value of {i} ",
                        f"probability of {i}.",
                        f"probability of {i} ",
                        f"is {i}.",
                        f"is {i}%",
                        f"is {i} ",
                        f"around {i}.",
                        f"around {i}%",
                        f"around {i} ",
                        f" {i}% chance",
                        f"be {i}.",
                        f"be {i},",
                        f"be {i} ",
                        f"at {i}.",
                        f"at {i},",
                        f"at {i} ",
                        f"approximately {i}.",
                        f"approximately {i} ",
                        f"as {i}.",
                        f"as {i} ",
                        f"= {i}.",
                        f" a {i} out of",
                    ]:
                        if parse in response:
                            if not prev_parses or parse not in prev_parses[-1]:
                                parsed_answer = i
                                prev_parses.append((parse, parsed_answer))

            if len(prev_parses) != 1:
                prev_parses = [pp for pp in prev_parses if pp[1] != 100]
            
            assert len(prev_parses) == 1
            parsed_answer = prev_parses[0][1]
        
        return float(parsed_answer)


class OpenAIChatReasoner(LLMReasoner):
    """'Chat' style OpenAI models like GPT-4 and GPT-3.5"""

    def __init__(self, model: str):
        super().__init__(model)
        openai.api_key = os.environ['OPENAI']
    
    @property
    def name(self) -> str:
        return f"{self.model}-openai-chat"

    def rate_e2_prompt(
        self, 
        prompt: List[Dict[str, str]], 
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> Optional[float]:

        completion = openai.ChatCompletion.create(
              model=self.model,
              messages=prompt,
              temperature=temperature,
            )
        assistant_message = completion.choices[0].message.content

        return self.parse_e2_response(assistant_message)
    

class OpenAICompletionReasoner(LLMReasoner):
    """'Completion' style OpenAI models like text-danvinci-003"""

    def __init__(self, model: str):
        super().__init__(model)
        openai.api_key = os.environ['OPENAI']
    
    @property
    def name(self) -> str:
        return f"{self.model}-openai-completion"

    def rate_e2_prompt(
        self, 
        prompt: str, 
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> Optional[float]:

        completion = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=100,
            logprobs=5,
        )

        return self.calculate_e2_rating(completion["choices"][0].logprobs.top_logprobs[-1])


class CohereCompletionReasoner(LLMReasoner):

    def __init__(self, model: str):
        super().__init__(model)
        self.co = cohere.Client(os.environ['COHERE'])
    
    @property
    def name(self) -> str:
        return f"{self.model}-cohere-completion"

    def rate_e2_prompt(
        self, 
        prompt: str, 
        temperature: int = config.DEFAULT_TEMPERATURE,
    ) -> Optional[float]:

        prediction = self.co.generate(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=100,
        )
        print(prediction)
        return self.parse_e2_response(prediction.generations[0].text)
    

class TextSynthCompletionReasoner(LLMReasoner):

    @property
    def name(self) -> str:
        return f"{self.model}-textsynth-completion"

    def rate_e2_prompt(
        self, 
        prompt: str, 
        temperature: int = 0.11,
    ) -> Optional[float]:
        api_url = "https://api.textsynth.com"
        response = requests.post(api_url + "/v1/engines/" + self.model + "/completions", headers = { "Authorization": "Bearer " + os.environ['TEXTSYNTH'] }, json = { "prompt": prompt, "max_tokens": 100, "temperature": temperature })
        resp = response.json()
        if "text" in resp: 
            return self.parse_e2_response(resp["text"])
        else:
            return None
