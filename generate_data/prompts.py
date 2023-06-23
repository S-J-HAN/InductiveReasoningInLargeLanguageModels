from dataclasses import dataclass

@dataclass(frozen=True)
class Experiment2Prompt:

    SYSTEM: str = "You are an expert on {} and the types of real world properties that they have. The questions you'll see don't have right or wrong answers, and you are willing to use your best judgment and commit to a concrete, specific response even in cases where you can't be sure that you are correct."
    CONTEXT: str = "We're going to show you a series of claims relating to {} and the properties they share. Rather than mention any specific property (e.g. \"Hyenas have sesamoid bones\") we'll refer to an unspecified property (e.g. \"Hyenas have property P\"). Each claim may be true or false, and to help you decide which, we'll provide you with facts about whether or not other {} have the same property (e.g. \"Lions have property P\", and \"Zebras have property P\").\n\nThis section contains a series of claims that include {}. Before we start, we'll give you two examples as practice.\n\n"
    ARGUMENT: str = "Argument A:\n{}\nClaim - {} have property P.\n\n"
    QUESTION: str = "Question: Given the facts presented, how likely is it that the claim is true?"
    OPTION: str = "Indicate your answer by providing a number between 0 and 100, where 0 means that the claim is very unlikely and 100 means that the claim is very likely."
    FACT: str = "Fact - {} have property P."
    SEGWAY: str = "Now that you've practiced you're ready to continue on to the main trials for this section.\n\n"