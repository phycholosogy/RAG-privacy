import fire
from llama import Llama
import warnings
import json
import os
"""
This file is to run large language model.
The running instructions have been generated in file f'{experiment name}.sh'
Please run the following command:
nohup bash {experiment name}.sh > output_name.out
or: bash {experiment name}.sh
"""


def main(
        ckpt_dir: str,       # LLM model name
        path: str,           # input and output place
        tokenizer_path: str = 'tokenizer.model',
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 4096,
        max_gen_len: int = 256,
        max_batch_size: int = 1,
):
    """
    Entry point of the program for generating text using a pretrained model
    :param
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        path (str): path to the experiment
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int): The maximum sequence length for input prompts. Defaults to 4096.
        max_gen_len (int): The maximum length of generated sequences. Defaults to 256.
        max_batch_size (int): The maximum batch size for generating sequences. Defaults to 1.
    """
    generator = Llama.build(
        ckpt_dir='Model/' + ckpt_dir,
        tokenizer_path='Model/' + tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print(path)
    if os.path.exists(f'./Inputs&Outputs/{path}/set.json'):
        print('summarizing now')
        # need to summarize
        with open(f'./Inputs&Outputs/{path}/set.json', "r") as file:
            settings = json.load(file)
        suf = settings['suffix']
        adh_1 = settings['adhesive_con']
        adh_2 = settings['adhesive_prompt']
        with open(f"./Inputs&Outputs/{path}/question.txt", 'r', encoding='utf-8') as f_que:
            all_questions = f_que.read()
        questions = all_questions.split('\n===================\n')[:-1]
        with open(f"./Inputs&Outputs/{path}/context.txt", 'r', encoding='utf-8') as f_con:
            all_contexts = f_con.read()
        contexts = all_contexts.split('\n===================\n')[:-1]
        su_1 = "Given the following question and context, extract any part of the context"\
               + " *AS IS* that is relevant to answer the question. If none of the context is relevant"\
               + " return NO_OUTPUT.\n\nRemember, *DO NOT* edit the extracted parts of the context.\n\n> Question: "
        su_2 = "\n> Context:\n>>>\n"
        su_3 = "\n>>>\nExtracted relevant parts:"
        k = len(contexts) // len(questions)
        prompt_ge_contexts = []
        summarize_contexts = []
        for i in range(len(questions)):
            ques = questions[i]
            for j in range(k):
                context = contexts[i*k+j]
                prompt_ge_context = su_1 + ques + su_2 + context + su_3
                prompt_ge_contexts.append(prompt_ge_context)
                results = generator.text_completion(
                    [prompt_ge_context],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                ans = results[0]['generation']
                summarize_contexts.append(ans)
        with open(f"./Inputs&Outputs/{path}/summarize_contexts.txt", 'w', encoding='utf-8') as f_c:
            for item in summarize_contexts:
                f_c.write(item + '\n===================\n')
        with open(f"./Inputs&Outputs/{path}/generate_summarize_prompt.txt", 'w', encoding='utf-8') as f_g:
            for item in prompt_ge_contexts:
                f_g.write(item + '\n===================\n')
        with open(f"./Inputs&Outputs/{path}/prompts.txt", 'w', encoding='utf-8') as f_p:
            for i in range(len(questions)):
                con_u = adh_1.join(summarize_contexts[i*k:i*k+k])
                prompt = suf[0] + con_u + adh_2 + suf[1] + questions[i] + adh_2 + suf[2]
                f_p.write(prompt + '\n===================\n')
    # generate output
    print('generating output')
    with open(f"./Inputs&Outputs/{path}/prompts.txt", 'r', encoding='utf-8') as f:
        all_prompts = f.read()
    all_prompts = all_prompts.split('\n===================\n')[:-1]
    with open(f"./Inputs&Outputs/{path}/outputs-{ckpt_dir[8:]}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}.txt",
              'w', encoding='utf-8') as file:
        for i in range(len(all_prompts)):
            prompt = all_prompts[i]
            prompts = [prompt]
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            file.write(f"{results[0]['generation']}")
            file.write("\n===================\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    fire.Fire(main)
