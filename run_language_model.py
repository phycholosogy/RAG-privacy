import fire
from llama import Llama
import warnings
import json
import os
from langchain_openai import OpenAI
"""
This file is to run large language model.
The running instructions have been generated in file f'{experiment name}.sh'
Please run the following command:
nohup bash {experiment name}.sh > output_name.out
or: bash {experiment name}.sh
"""
# If you want to use OpenAI's model, please set API here
os.environ['OPENAI_API_KEY'] = 'YOUR API KEY'


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
        ckpt_dir: The directory containing checkpoint files for the pretrained model.
            If you want to use gpt, please trans the true name of a gpt model.
            If you just input 'gpt', we provide default model: 'gpt-3.5-turbo-instruct'.
            For security reasons, please type your openai API inside this file
        path: path to the experiment
        tokenizer_path: The path to the tokenizer model used for text encoding/decoding.
        temperature: The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p: The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len: The maximum sequence length for input prompts. Defaults to 4096.
        max_gen_len: The maximum length of generated sequences. Defaults to 256.
        max_batch_size: The maximum batch size for generating sequences. Defaults to 1.
    :return:
        the output will be saved at
            f"./Inputs&Outputs/{path}/outputs-{ckpt_dir}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}.json"
    """
    print(path)
    generator = None
    llm = None
    # summary stage
    if os.path.exists(f'./Inputs&Outputs/{path}/set.json'):
        flag_llm = 'llama'
        print('summarizing now')
        # need to summarize
        with open(f'./Inputs&Outputs/{path}/set.json', "r") as file:
            settings = json.load(file)
        summary_model = settings['infor']
        para_flag = False
        if summary_model.find('-para'):
            para_flag = True
            summary_model = summary_model.strip('-para')
        if summary_model.find('gpt') != -1:
            flag_llm = 'gpt'
            if summary_model == 'gpt':
                summary_model = 'gpt-3.5-turbo-instruct'
            llm = OpenAI(model=summary_model, temperature=temperature, top_p=top_p, max_tokens=max_gen_len)
        else:
            generator = Llama.build(
                ckpt_dir='Model/' + summary_model,
                tokenizer_path='Model/' + tokenizer_path,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
            )

        suf = settings['suffix']
        adh_1 = settings['adhesive_con']
        adh_2 = settings['adhesive_prompt']
        with open(f"./Inputs&Outputs/{path}/question.json", 'r', encoding='utf-8') as f_que:
            questions = json.loads(f_que.read())
        with open(f"./Inputs&Outputs/{path}/context.txt", 'r', encoding='utf-8') as f_con:
            contexts = json.loads(f_con.read())

        su_1 = "Given the following question and context, extract any part of the context" \
               + " *AS IS* that is relevant to answer the question. If none of the context is relevant" \
               + " return NO_OUTPUT.\n\nRemember, *DO NOT* edit the extracted parts of the context.\n\n> Question: "
        if para_flag:
            su_1 = "Given the following question and context, extract any part of the context" \
                   + " *AS IS* that is relevant to answer the question. If none of the context is relevant" \
                   + " return NO_OUTPUT.\n\n> Question: "
        su_2 = "\n> Context:\n>>>\n"
        su_3 = "\n>>>\nExtracted relevant parts:"
        prompt_ge_contexts = []
        summarize_contexts = []
        for i in range(len(questions)):
            ques = questions[i]
            k_contexts = contexts[i]
            ge_contexts = []
            sum_contexts = []
            for j in range(len(k_contexts)):
                context = k_contexts[j]
                prompt_ge_context = su_1 + ques + su_2 + context + su_3
                ge_contexts.append(prompt_ge_context)
                if flag_llm == 'gpt':
                    ans = llm.invoke(prompt_ge_context)
                else:
                    results = generator.text_completion(
                        [prompt_ge_context],
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    ans = results[0]['generation']
                sum_contexts.append(ans)
            summarize_contexts.append(sum_contexts)
            prompt_ge_contexts.append(ge_contexts)

        with open(f"./Inputs&Outputs/{path}/summarize_contexts.json", 'w', encoding='utf-8') as f_c:
            f_c.write(json.dumps(summarize_contexts))
        with open(f"./Inputs&Outputs/{path}/generate_summarize_prompt.json", 'w', encoding='utf-8') as f_g:
            f_g.write(json.dumps(prompt_ge_contexts))
        prompts = []
        for i in range(len(questions)):
            con_u = adh_1.join(summarize_contexts[i])
            prompt = suf[0] + con_u + adh_2 + suf[1] + questions[i] + adh_2 + suf[2]
            prompts.append(prompt)
        with open(f"./Inputs&Outputs/{path}/prompts.txt", 'w', encoding='utf-8') as f_p:
            f_p.write(json.dumps(prompts))

    flag_llm = 'llama'
    if ckpt_dir.find('gpt') != -1:
        # Type your API here
        if ckpt_dir == 'gpt':
            ckpt_dir = 'gpt-3.5-turbo-instruct'
        llm = OpenAI(model=ckpt_dir, temperature=temperature, top_p=top_p, max_tokens=max_gen_len)
        flag_llm = 'gpt'
    else:
        generator = Llama.build(
            ckpt_dir='Model/' + ckpt_dir,
            tokenizer_path='Model/' + tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
    # generate output
    print('generating output')
    with open(f"./Inputs&Outputs/{path}/prompts.json", 'r', encoding='utf-8') as f:
        all_prompts = json.loads(f.read())
    answer = []
    for i in range(len(all_prompts)):
        if flag_llm == 'gpt':
            ans = llm.invoke(all_prompts[i])
        else:
            results = generator.text_completion(
                [all_prompts[i]],
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            ans = results[0]['generation']
        answer.append(ans)

    with open(
            f"./Inputs&Outputs/{path}/outputs-{ckpt_dir}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}.json",
            'w', encoding='utf-8') as file:
        file.write(json.dumps(answer))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    fire.Fire(main)
