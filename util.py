import random

def label(q):
    if q['is_human'] == "Yes":
        return "human"
    return "llm"

def create_selection_question(q):
    if q['human_index'] == 1:
        message_1 = q['human_message']
        message_2 = q['message']
    else:
        message_1 = q['message']
        message_2 = q['human_message']
    return f"Given the two messages, which one of them was created by human and other by LLM, your task is to find the message that was wrote by human and not LLM. Provide a full explanation \n message (1): {message_1} \n message (2): {message_2} \nA: the message wrote by human is message ("


