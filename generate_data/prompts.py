from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import config

E1_OPTIONS = [
    "A - Argument A is much stronger",
    "B - Argument A is stronger",
    "C - Argument A is slightly stronger",
    "D - Argument B is slightly stronger",
    "E - Argument B is stronger",
    "F - Argument B is much stronger",
]

ChatMessage = List[Dict[str, str]]

@dataclass(frozen=True)
class Argument:
    premises: List
    conclusion: str


class Experiment1Prompt:

    SYSTEM: str = "You are an expert on {} and the types of real world properties that they have. The questions you'll see don't have right or wrong answers, and you are willing to use your best judgment and commit to a concrete, specific response even in cases where you can't be sure that you are correct."
    CONTEXT: str = "We are interested in how people evaluate arguments. On each trial there will be two arguments labeled 'A' and 'B'. Each will contain one, two, or three statements separated from a claim by a line. Assume that the statements above the line are facts, and choose the argument whose facts provide a better reason for believing the claim. These are subjective judgments; there are no right or wrong answers."
    ARGUMENT: str = "Argument {}:\n{}\nClaim - {} have property P."
    FACT: str = "Fact - {} have property P."
    QUESTION: str = "Question: Assuming all the facts given are true, which argument makes a stronger case for the claim? To get the best answer, first write down your reasoning. Then, based on this, indicate the strength of your preference by providing one of the following options:"
    OPTION: str = "\n".join(E1_OPTIONS)

    def generate_prompt(
        self,
        argument_a: Argument,
        argument_b: Argument,
        domain: str,
        is_completion: bool,
    ) -> str | ChatMessage:
        if is_completion:
            return self._generate_completion_prompt(argument_a, argument_b, domain)
        else:
            return self._generate_chat_prompt(argument_a, argument_b, domain)
    
    def _fill_prompts(
        self,
        argument_a: Argument,
        argument_b: Argument,
        domain: str,
    ) -> Tuple[str,str,str]:
        
        parent_domain = config.PARENT_DOMAINS[domain].lower()
        system_prompt = self.SYSTEM.format(parent_domain)
        argument_a_prompt = self.ARGUMENT.format("A", "\n".join([self.FACT.format(p) for p in argument_a.premises]), argument_a.conclusion)
        argument_b_prompt = self.ARGUMENT.format("B", "\n".join([self.FACT.format(p) for p in argument_b.premises]), argument_b.conclusion)
        
        return system_prompt, argument_a_prompt, argument_b_prompt

    def _generate_completion_prompt(
        self,
        argument_a: Argument,
        argument_b: Argument,
        domain: str,
    ) -> str:
        
        system_prompt, argument_a_prompt, argument_b_prompt = self._fill_prompts(argument_a, argument_b, domain)

        return "\n\n".join([system_prompt, self.CONTEXT, argument_a_prompt, argument_b_prompt, self.QUESTION + "\n" + self.OPTION])
    
    def _generate_chat_prompt(
        self,
        argument_a: Argument,
        argument_b: Argument,
        domain: str,  
    ) -> ChatMessage:
        
        system_prompt, argument_a_prompt, argument_b_prompt = self._fill_prompts(argument_a, argument_b, domain)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join([self.CONTEXT, argument_a_prompt, argument_b_prompt, self.QUESTION + "\n" + self.OPTION])},
        ]
        

class Experiment2Prompt:

    SYSTEM: str = "You are an expert on {} and the types of real world properties that they have. The questions you'll see don't have right or wrong answers, and you are willing to use your best judgment and commit to a concrete, specific response even in cases where you can't be sure that you are correct."
    CONTEXT: str = "We're going to show you a series of claims relating to {} and the properties they share. Rather than mention any specific property (e.g. \"Hyenas have sesamoid bones\") we'll refer to an unspecified property (e.g. \"Hyenas have property P\"). Each claim may be true or false, and to help you decide which, we'll provide you with facts about whether or not other {} have the same property (e.g. \"Lions have property P\", and \"Zebras have property P\")."
    TUTORIAL: str = "This section contains a series of claims that include {}. Before we start, we'll give you two examples as practice.\n\n"
    ARGUMENT: str = "Argument A:\n{}\nClaim - {} have property P."
    FACT: str = "Fact - {} have property P."
    QUESTION: str = "Question: Given the facts presented, how likely is it that the claim is true?"
    OPTION: str = "Indicate your answer by providing a number between 0 and 100, where 0 means that the claim is very unlikely and 100 means that the claim is very likely."
    SEGWAY: str = "Now that you've practiced you're ready to continue on to the main trials for this section."

    def generate_prompt(
            self,
            argument: Argument,
            domain: str,
            is_completion: bool,
            tutorial_prompt: Optional[str | ChatMessage] = None,
    ) -> str | ChatMessage:
        if is_completion:
            return self._generate_completion_prompt(argument, domain, tutorial_prompt)
        else:
            return self._generate_chat_prompt(argument, domain, tutorial_prompt)
        
    def _fill_prompts(
        self,
        argument: Argument,
        domain: str,
    ) -> Tuple[str,str,str]:
        argument_prompt = self.ARGUMENT.format("\n".join([self.FACT.format(p) for p in argument.premises]), argument.conclusion)
        parent_domain = config.PARENT_DOMAINS[domain].lower()
        system_prompt = self.SYSTEM.format(parent_domain)
        context_prompt = self.CONTEXT.format(parent_domain,parent_domain)
        return argument_prompt, system_prompt, context_prompt

    def _generate_completion_prompt(
            self,
            argument: Argument,
            domain: str,
            tutorial_prompt: str = "",
    ) -> str:
        
        argument_prompt, system_prompt, context_prompt = self._fill_prompts(argument, domain)
        if tutorial_prompt:
            return tutorial_prompt + argument_prompt + "\n\n" + self.QUESTION + " " + self.OPTION
        else:
            return system_prompt + "\n\n" + context_prompt + "\n\n" + argument_prompt + "\n\n" + self.QUESTION + " " + self.OPTION
    
    def _generate_chat_prompt(
        self,
        argument: Argument,
        domain: str,
        tutorial_prompt: ChatMessage = [],
    ) -> ChatMessage:
        
        argument_prompt, system_prompt, context_prompt = self._fill_prompts(argument, domain)
        if tutorial_prompt:
            return tutorial_prompt + [{"role": "user", "content": argument_prompt + "\n\n" + self.QUESTION + " " + self.OPTION}]
        else:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_prompt + "\n\n" + argument_prompt + "\n\n" + self.QUESTION + " " + self.OPTION},
            ]
        

class SimilarityPrompt:

    QUESTION: str = "You are an expert on {} and the different properties that they have. With these properties in mind, we will ask you whether two {} are similar or not.\n\nQuestion: With their respective properties in mind, how similar are {} and {} on a scale of 0 to 20? Answer with a single number.\nAnswer:"

    def generate_prompt(self, domain: str, category1: str, category2: str, is_completion: bool) -> str | ChatMessage:
        if is_completion:
            return self._generate_completion_prompt(domain, category1, category2)
        else:
            return self._generate_chat_prompt(domain, category1, category2)
    
    def _generate_completion_prompt(self, domain: str, category1: str, category2: str) -> str:
        return self.QUESTION.format(domain.lower(), domain.lower(), category1.lower(), category2.lower())

    def _generate_chat_prompt(self, domain: str, category1: str, category2: str) -> ChatMessage:
        return [{"role": "user", "content": self.QUESTION.format(domain.lower(), domain.lower(), category1.lower(), category2.lower())}]
