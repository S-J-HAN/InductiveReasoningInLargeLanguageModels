# File locations
DATA = "../data"
DEDEYNE_DATA = f"{DATA}/dedeyne"
E1_DATA = f"{DATA}/experiment_1"
E2_DATA = f"{DATA}/experiment_2"
SIMILARITY_DATA = f"{DATA}/similarity"


# Default LLM API temperature
DEFAULT_TEMPERATURE = 0


# Sleep times to try to minimise exceeding RPS limits
SLEEP_TIMES = {
    "cohere": 15,
    "openai": 1,
    "textsynth": 1,
}


# Some experiment 2 test prompts
TEST_PROMPT = "You are an expert on objects and the types of real world properties that they have. The questions you'll see don't have right or wrong answers, and you are willing to use your best judgment and commit to a concrete, specific response even in cases where you can't be sure that you are correct.\n\nWe're going to show you a series of claims relating to objects and the properties they share. Rather than mention any specific property (e.g. 'Hyenas have sesamoid bones') we'll refer to an unspecified property (e.g. 'Hyenas have property P'). Each claim may be true or false, and to help you decide which, we'll provide you with facts about whether or not other objects have the same property (e.g. 'Lions have property P', and 'Zebras have property P').\n\nArgument A:\nFact - Airplanes have property P.\nFact - Helicopters have property P.\nClaim - All vehicles have property P.\n\nQuestion: Given the facts presented, how likely is it that the claim is true? Indicate your answer by providing a number between 0 and 100, where 0 means that the claim is very unlikely and 100 means that the claim is very likely."
TEST_MESSAGE = [{'role': 'system', 'content': "You are an expert on objects and the types of real world properties that they have. The questions you'll see don't have right or wrong answers, and you are willing to use your best judgment and commit to a concrete, specific response even in cases where you can't be sure that you are correct."}, {'role': 'user', 'content': 'We\'re going to show you a series of claims relating to objects and the properties they share. Rather than mention any specific property (e.g. "Hyenas have sesamoid bones") we\'ll refer to an unspecified property (e.g. "Hyenas have property P"). Each claim may be true or false, and to help you decide which, we\'ll provide you with facts about whether or not other objects have the same property (e.g. "Lions have property P", and "Zebras have property P").\n\nThis section contains a series of claims that include two supporting facts. Before we start, we\'ll give you two examples as practice.\n\nArgument A:\nFact - Lemons have property P.\nFact - Limes have property P.\nClaim - All fruits have property P.\n\nQuestion: Given the facts presented, how likely is it that the claim is true? Indicate your answer by providing a number between 0 and 100, where 0 means that the claim is very unlikely and 100 means that the claim is very likely.'}, {'role': 'assistant', 'content': '60'}, {'role': 'user', 'content': 'Argument A:\nFact - Bananas have property P.\nFact - Watermelons have property P.\nClaim - All fruits have property P.\n\nQuestion: Given the facts presented, how likely is it that the claim is true? Indicate your answer by providing a number between 0 and 100, where 0 means that the claim is very unlikely and 100 means that the claim is very likely.'}, {'role': 'assistant', 'content': '65'}, {'role': 'user', 'content': "Now that you've practiced you're ready to continue on to the main trials for this section.\n\nArgument A:\nFact - Airplanes have property P.\nFact - Buses have property P.\nClaim - All vehicles have property P.\n\nQuestion: Given the facts presented, how likely is it that the claim is true? Indicate your answer by providing a number between 0 and 100, where 0 means that the claim is very unlikely and 100 means that the claim is very likely."}]


# Map Dedeyne domains to higher level domains
PARENT_DOMAINS = {
    "Mammals": "living things",
    "Birds": "living things",
    "Vehicles": "objects",
}