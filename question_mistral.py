"""Mistral questioning, this file contains a lot of code that deals with creating consistency with the evalharness evaluation pipeline."""

import requests
import time
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

SPLITS = {'validation': 'validation.csv',
          'test': 'llmvshuman_test_corrected.csv'}
FEW_SHOT_EXAMPLES_DF = pd.read_csv(
    "hf://datasets/tomshe/turing_phishing_selection/" + SPLITS["validation"])
EXAMPLES_DF = pd.read_csv(
    "hf://datasets/tomshe/turing_phishing_selection/" + SPLITS["test"])

PAIRED_Q_PROMPT = """
Given the two messages, which one of them was created by a human and the other
 by LLM, your task is to find the message that was written by a human and not LLM.
 Provide a full explanation
 message (1): {message_1}
 message (2): {message_2}
 A: the message written by human is message (
"""
SINGLE_Q_PROMPT = """
 Your task is to Identify if a large language model or human created this message:
 {message}
 Identify if a large language model or human created this message: [LLM] / [human]
 A:[{answer}]
"""


# Initialize Mistral API parameters
# Mistral API key (need to insert to use it!)
API_KEY = ""
# Mistral API endpoint
API_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"


def _build_single_fewshot_prompt_prefix(examples_df):
    prompt = ''
    for example in examples_df.iterrows():
        example = example[1]
        current = SINGLE_Q_PROMPT.format(
            message=example['message'], answer='LLM' if example['target'] == 'llm' else 'human')
        prompt += current
        prompt += '\n\n'
    return prompt


def _build_paired_fewshot_prompt_prefix(examples_df):
    # To maintaine consistency with other LLM questioning.
    prefix_ordering = [('quigley', 'human'), ('gallant', 'llm'),
                       ('hille', 'human'), ('morgan', 'llm')]
    prompt = ""
    for name, first_example_target in prefix_ordering:
        person_entries = examples_df[examples_df['person'].str.lower(
        ).str.contains(name, na=False)]
        ordered_examples = (
            person_entries[person_entries['target'] ==
                           first_example_target]['message'].to_string(index=False).replace('\n', ' '),
            person_entries[person_entries['target'] != first_example_target]['message'].to_string(index=False).replace('\n', ' '))
        prompt += PAIRED_Q_PROMPT.format(
            message_1=ordered_examples[0], message_2=ordered_examples[1], answer='1' if first_example_target == 'human' else '2')
    return prompt


def _create_paired_prompts_list(paired_prompt_prefix):
    fewshot_prompts, zero_prompts, answers, mesages_src = [], [], [], []
    for group in EXAMPLES_DF.groupby('person'):
        messages = group[1]['message']
        fewshot_prompts.append(paired_prompt_prefix + PAIRED_Q_PROMPT.format(
            message_1=messages.iloc[0], message_2=messages.iloc[1], answer="")[:-1])
        zero_prompts.append(PAIRED_Q_PROMPT.format(
            message_1=messages.iloc[0], message_2=messages.iloc[1], answer="")[:-1])
        answers.append('1' if group[1].iloc[0]['target'] == 'human' else '2')
        mesages_src.append(group[1].iloc[0]['model'] if pd.notna(
            group[1].iloc[0]['model']) else group[1].iloc[1]['model'])
    return fewshot_prompts, zero_prompts, answers, mesages_src


def complete_texts(prompt):
    """
    Completes the given prompts using Mistral's API.

    Args:
        prompts (list): A list of strings, each representing a prompt to complete.

    Returns:
        list: A list of text completions corresponding to the prompts.
    """
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": 'mistral-large-latest',
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "min_tokens": 0,
        'temperature': 0.7,
        "top_p": 1.0,
        "stop": ["\n\n\n"]
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]
    except requests.exceptions.RequestException as e:
        print(f"Error completing prompts: \n{e}")
        return None


def _process_single_questions_completion(completion):
    completion = completion.lower()

    if 'llm' in completion:
        if 'human' in completion:
            return "None"
        else:
            return "llm"
    else:
        if 'human' in completion:
            return "human"
        else:
            return "None"


def _process_paired_questions_completion(completion):
    completion = completion.lower()

    if '1' in completion or 'one' in completion:
        if '2' in completion or 'two' in completion:
            return 'None'
        else:
            return '1'
    else:
        if '2' in completion or 'two' in completion:
            return '2'
        else:
            return 'None'


def _get_completions(prompts,  processing_func):
    completions, processed_completions = [], []
    for i, prompt in enumerate(prompts):
        completion = complete_texts(prompt)
        completions.append(completion['message']['content'])
        processed_completion = processing_func(
            completion['message']['content'])
        processed_completions.append([processed_completion])
        # To correspond to the free-tier QPS allowance
        time.sleep(2.0)
    return completions, processed_completions


# Example usage
if __name__ == "__main__":

    SINGLE_QUESTIONS_PREFIX = _build_single_fewshot_prompt_prefix(
        FEW_SHOT_EXAMPLES_DF)
    PAIRED_QUESTIONS_PREFIX = _build_paired_fewshot_prompt_prefix(
        FEW_SHOT_EXAMPLES_DF)

    # Create prompts, answers, sources per dataset entry
    fewshot_single_prompts = [SINGLE_QUESTIONS_PREFIX + SINGLE_Q_PROMPT.format(
        message=row[1]['message'], answer="")[:-1] for row in EXAMPLES_DF.iterrows()]
    zeroshot_single_prompts = [SINGLE_Q_PROMPT.format(
        message=row[1]['message'], answer="")[:-1] for row in EXAMPLES_DF.iterrows()]
    single_answers = [row[1]['target'] for row in EXAMPLES_DF.iterrows()]
    single_messages_src = [
        row[1]['model'] if pd.notna(row[1]['model']) else 'human' for row in EXAMPLES_DF.iterrows()
    ]
    fewshot_paired_prompts, zeroshot_paired_prompts, paired_answers, paired_messages_src = _create_paired_prompts_list(
        PAIRED_QUESTIONS_PREFIX)

    # Get completions in both setups
    fewshot_single_completions, fewshot_single_processed_completions = _get_completions(
        fewshot_single_prompts, _process_single_questions_completion)
    fewshot_paired_completions, fewshot_paired_processed_completions = _get_completions(
        fewshot_paired_prompts, _process_paired_questions_completion)

    mistral_fewshot_answers = pd.DataFrame(data={
        "Prompt": fewshot_single_prompts + fewshot_paired_prompts,
        'Completions': fewshot_single_completions + fewshot_paired_completions,
        'filtered_resps': fewshot_single_processed_completions + fewshot_paired_processed_completions,
        'target': single_answers + paired_answers,
        'doc': single_messages_src + paired_messages_src
    })

    zeroshot_single_completions, zeroshot_single_processed_completions = _get_completions(
        zeroshot_single_prompts, _process_single_questions_completion)
    zeroshot_paired_completions, zeroshot_paired_processed_completions = _get_completions(
        zeroshot_paired_prompts, _process_paired_questions_completion)


    mistral_zeroshot_answers = pd.DataFrame(data={
        "Prompt": zeroshot_single_prompts + zeroshot_paired_prompts,
        'Completions': zeroshot_single_completions + zeroshot_paired_completions,
        'filtered_resps': zeroshot_single_processed_completions + zeroshot_paired_processed_completions,
        'target': single_answers + paired_answers,
        'doc': single_messages_src + paired_messages_src
    })

    mistral_fewshot_answers.to_csv('./data/mistral_answers/mistral_fewshot.csv')
    mistral_zeroshot_answers.to_csv(
        './data/mistral_answers/mistral_zeroshot.csv')
