from typing import List, Dict, Any, Optional

import config


class Experiment2Prompt:

    SYSTEM: str = "You are an expert on {} and the types of real world properties that they have. The questions you'll see don't have right or wrong answers, and you are willing to use your best judgment and commit to a concrete, specific response even in cases where you can't be sure that you are correct."
    CONTEXT: str = "We're going to show you a series of claims relating to {} and the properties they share. Rather than mention any specific property (e.g. \"Hyenas have sesamoid bones\") we'll refer to an unspecified property (e.g. \"Hyenas have property P\"). Each claim may be true or false, and to help you decide which, we'll provide you with facts about whether or not other {} have the same property (e.g. \"Lions have property P\", and \"Zebras have property P\")."
    TUTORIAL: str = "This section contains a series of claims that include {}. Before we start, we'll give you two examples as practice.\n\n"
    ARGUMENT: str = "Argument A:\n{}\nClaim - {} have property P."
    QUESTION: str = "Question: Given the facts presented, how likely is it that the claim is true?"
    OPTION: str = "Indicate your answer by providing a number between 0 and 100, where 0 means that the claim is very unlikely and 100 means that the claim is very likely."
    FACT: str = "Fact - {} have property P."
    SEGWAY: str = "Now that you've practiced you're ready to continue on to the main trials for this section."


    def generate_prompt(
            self,
            premises: List[str],
            conclusion: str,
            domain: str,
            is_completion: bool,
            tutorial_prompt: Optional[Any] = None,
    ) -> Any:
        if is_completion:
            return self._generate_completion_prompt(premises, conclusion, domain, tutorial_prompt)
        else:
            return self._generate_chat_prompt(premises, conclusion, domain, tutorial_prompt)


    def _generate_completion_prompt(
            self,
            premises: List[str],
            conclusion: str,
            domain: str,
            tutorial_prompt: str = "",
    ) -> str:
        
        argument_prompt = self.ARGUMENT.format("\n".join([self.FACT.format(p) for p in premises]), conclusion)
        if tutorial_prompt:
            return tutorial_prompt + argument_prompt + "\n\n" + self.QUESTION + " " + self.OPTION
        else:

            parent_domain = config.PARENT_DOMAINS[domain].lower()
            system_prompt = self.SYSTEM.format(parent_domain)
            context_prompt = self.CONTEXT.format(parent_domain,parent_domain)

            return system_prompt + "\n\n" + context_prompt + "\n\n" + argument_prompt + "\n\n" + self.QUESTION + " " + self.OPTION
    

    def _generate_chat_prompt(
        self,
        premises: List[str],
        conclusion: str,
        domain: str,
        tutorial_prompt: List[Dict[str,str]] = [],
    ) -> config.ChatMessage:
        
        argument_prompt = self.ARGUMENT.format("\n".join([self.FACT.format(p) for p in premises]), conclusion)
        if tutorial_prompt:
            return tutorial_prompt + [{"role": "user", "content": argument_prompt + "\n\n" + self.QUESTION + " " + self.OPTION}]
        else:

            parent_domain = config.PARENT_DOMAINS[domain].lower()
            system_prompt = self.SYSTEM.format(parent_domain)
            context_prompt = self.CONTEXT.format(parent_domain,parent_domain)

            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_prompt + "\n\n" + argument_prompt + "\n\n" + self.QUESTION + " " + self.OPTION},
            ]
        

class SimilarityPrompt:

    QUESTION: str = "You are an expert on {} and the different properties that they have. With these properties in mind, we will ask you whether two {} are similar or not.\n\nQuestion: With their respective properties in mind, how similar are {} and {} on a scale of 0 to 20? Answer with a single number.\nAnswer:"

    def generate_prompt(self, domain: str, category1: str, category2: str, is_completion: bool) -> Any:
        if is_completion:
            return self._generate_completion_prompt(domain, category1, category2)
        else:
            return self._generate_chat_prompt(domain, category1, category2)
    

    def _generate_completion_prompt(self, domain: str, category1: str, category2: str) -> str:
        return self.QUESTION.format(domain.lower(), domain.lower(), category1.lower(), category2.lower())


    def _generate_chat_prompt(self, domain: str, category1: str, category2: str) -> config.ChatMessage:
        return [{"role": "user", "content": self.QUESTION.format(domain.lower(), domain.lower(), category1.lower(), category2.lower())}]
