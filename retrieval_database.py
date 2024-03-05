"""
This file is the code about retrieval database part.
In this file, we support functions to build a vector database for retrieval, and provide function to load the database.
It contains following functions:
1. pre_process_dataset:
    pre precess the dataset for construction of retrieval database
2. split_dataset:
    split the dataset to the train set and test set
2. construct_retrieval_database:
    construct a retrieval database
3. load_retrieval_database_from_address
    load a pre-built retrieval database based on a secure address
4. load_retrieval_database_from_parameter
    load a pre-built retrieval database based on the database name and construct method.
    if you use the construct_retrieval_database to build a retrieval database,
    this function will help you access the database clearer
It contains utilitys, these are internal calls between functions, you can skip these functions:
1. find_all_file: find all files in folder f'{path}'
2. get_encoding_of_file: get the encoding of the file
3. get_embed_model: get the embedding model
"""
import os
import random
import shutil
import json
import argparse
from typing import List
from chardet.universaldetector import UniversalDetector

import torch
import langchain
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from nltk.tokenize import RegexpTokenizer


def find_all_file(path: str) -> List[str]:
    """
    return the list of all files of a folder
    :param:
        path: the path of the folder
    :return:
        A list containing the paths of all files in the folder
    """
    for root, ds, fs in os.walk(path):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


def get_encoding_of_file(path: str) -> str:
    """
    return the encoding of a file
    """
    detector = UniversalDetector()
    with open(path, 'rb') as file:
        data = file.readlines()
        for line in data:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    return detector.result['encoding']


def get_embed_model(encoder_model_name: str,
                    device: str = 'cpu',
                    retrival_database_batch_size: int = 256) -> OpenAIEmbeddings:
    """
    get embedding model
    You can also code for other embedding model
    :param:
        encoder_model_name: name of encoder model. Options:
            open-ai: OpenAI Embeddings
            all-MiniLM-L6-v2: default Embedding method
            bge-large-en-v1.5: bge-large-en-v1.5 from BAAI
            e5-base-v2： e5-base-v2 from intfloat
        device: cpu or gpu if available
        retrival_database_batch_size: batch size
    :return:
        the embedding model
    """
    if encoder_model_name == 'open-ai':
        embed_model = OpenAIEmbeddings()
    elif encoder_model_name == 'all-MiniLM-L6-v2':
        embed_model = HuggingFaceEmbeddings(
            model_name=encoder_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': retrival_database_batch_size},
        )
    elif encoder_model_name == 'bge-large-en-v1.5':
        embed_model = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5',
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': retrival_database_batch_size}
        )
    elif encoder_model_name == 'e5-base-v2':
        embed_model = HuggingFaceEmbeddings(
            model_name='intfloat/e5-base-v2',
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': retrival_database_batch_size}
        )
    else:
        try:
            embed_model = HuggingFaceEmbeddings(
                model_name=encoder_model_name,
                model_kwargs={'device': device},
                encode_kwargs={'device': device, 'batch_size': retrival_database_batch_size},
            )
        except encoder_model_name:
            raise Exception(f"Encoder {encoder_model_name} not found, please check.")
    return embed_model


def pre_process_dataset(data_name: str, change_method: str = 'body') -> None:
    """
    Preprocess the dataset for the retrieval database.
    You can write your own preprocessing function in this part
    :param
    `   data_name: name of the origin data
        change_method: used for enron email
            'body': only remain the body of the email
            'strip': delete the '\n' in the email, but remain other message
    :function
        pre_process_chatdoctor: how we pre-process the chatdoctor dataset
        pre_process_enron_mail: how we pre-process the enron mail dataset
    """

    data_store_path = 'Data'

    def pre_process_chatdoctor() -> None:
        """
        delete the instruction, the instruction is fixed as following:
            "If you are a doctor, please answer the medical questions based on the patient's description."
        In a retrieval dataset, the instruction is in no need.
        """
        file_path = os.path.join(data_store_path, 'chatdoctor200k/chatdoctor200k.json')
        with open(file_path, 'r') as f:
            content = f.read()
            data = json.loads(content)
        output_path = os.path.join(data_store_path, 'chatdoctor/chatdoctor.txt')
        with open(output_path, 'w', encoding="utf-8") as f:
            max_len = 0
            for i, item in enumerate(data):
                s = 'input: ' + item['input'] + '\n' + 'output: ' + item['output']
                s = s.replace('\xa0', ' ')
                if i != len(data) - 1:
                    s += '\n\n'
                max_len = max(max_len, len(s))
                f.write(s)
        print(f'Number of chatdoctor dataset is {len(data)}')  # 207408
        print(f'Max length of chatdoctor dataset is {max_len}')  # 11772

    def pre_process_enron_mail() -> None:
        num_file = 0
        data_path = os.path.join(data_store_path, data_name)
        for file_name in find_all_file(data_path):
            # detect the encode method of files:
            encoding = get_encoding_of_file(file_name)
            # load the data
            with open(file_name, 'r', encoding=encoding) as file:
                data = file.read()
            content = data.split('\n\n')
            new_content = ""
            for item in content:
                item_ = item.strip()
                if item_ == '':
                    continue
                if change_method == 'body':
                    num_other_title = 0
                    other_messages = ["Message-ID:", "Date:", "From:", "To:", "Subject:", "Mime-Version:", "X-Origin:",
                                      "Cc:", "Content-Transfer-Encoding:", "X-From:", "X-To:", "X-cc:", "X-bcc:",
                                      "Sent:", "X-Folder:", "X-FileName:", "Content-Type:", "Bcc:", "X-Origin:",
                                      "X-FileName:"]
                    for other_message in other_messages:
                        num_other_title += item_.count(other_message)
                    if num_other_title < 3:
                        new_content += item_.replace('\n', ' ')
                elif change_method == 'strip':
                    new_content += item_.replace('\n', ' ')
                new_content = new_content.strip()
                if new_content != "" and new_content[-1] != '.' and new_content[-1] != '?' and new_content[-1] != '!':
                    new_content += '.'
            if len(new_content) != 0:
                path = f'Data/enron-mail-{change_method}/' + file_name[16:] + '.txt'
                num_file += 1
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
        print(f'{data_name}-{change_method} num of files: {num_file}')

    if data_name == "chatdoctor200k":
        pre_process_chatdoctor()
    elif data_name == "enron-mail":
        pre_process_enron_mail()


def split_dataset(data_name: str, split_ratio: int = 0.99, num_eval: int = 1000, max_que_len: int = 50) -> None:
    """
    split the dataset to the train set and the test set
    :param:
        data_name: name of the dataset
        split_ratio: ratio of the train set
        num_eval: the number of samples from test sets to evaluate the performance
        max_que_len: max length of the input of the evaluation for enron-mail
    the train-set and the test-set will be stored at folder {data_name}-train and {data_name}-test
    """
    data_store_path = 'Data'
    if data_name == 'chatdoctor':
        with open('Data/chatdoctor/chatdoctor.txt', 'r', encoding="utf-8") as f:
            data = f.read()
        data = data.split('\n\n')
        output_train_path = os.path.join(data_store_path, 'chatdoctor-train/chatdoctor.txt')
        output_test_path = os.path.join(data_store_path, 'chatdoctor-test/chatdoctor.txt')
        num_ = int(split_ratio * len(data))
        random.shuffle(data)
        with open(output_train_path, 'w', encoding="utf-8") as f:
            f.write('\n\n'.join(data[:num_]))
        with open(output_test_path, 'w', encoding="utf-8") as f:
            f.write('\n\n'.join(data[num_:]))
        # getting information of performance evaluation
        test_data = data[num_:]
        random.shuffle(test_data)
        eval_data = test_data[:num_eval]
        eval_input = []
        eval_output = []
        for e_data in eval_data:
            item = e_data.split('\noutput: ')
            eval_input.append(item[0][7:])
            eval_output.append(item[1])
        with open(f'Data/{data_name}-test/eval_input.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(eval_input))
        with open(f'Data/{data_name}-test/eval_output.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(eval_output))
    else:
        """
        If a dataset is stored in multiple files, and you only want to partition the dataset at the file level
        You can use this code directly
        Alternatively, you can modify this section of the code according to your specific requirements.
        """
        data_path = os.path.join(data_store_path, data_name)
        all_file = []
        for file_name in find_all_file(data_path):
            all_file.append(file_name)
        random.shuffle(all_file)
        num_train = int(len(all_file) * split_ratio)
        train_all_file = all_file[:num_train]
        test_all_file = all_file[num_train:]
        print(f'Number of the training set is {len(train_all_file)}, number of the test set is {len(test_all_file)}')
        for train_file in train_all_file:
            source_file = train_file    # source path
            target_file = train_file.replace(data_name, f'{data_name}-train')
            if not os.path.exists(os.path.dirname(target_file)):
                os.makedirs(os.path.dirname(target_file))
            # using shutil to copy file
            shutil.copy2(source_file, target_file)

        for test_file in test_all_file:
            source_file = test_file    # source path
            target_file = test_file.replace(data_name, f'{data_name}-test')
            if not os.path.exists(os.path.dirname(target_file)):
                os.makedirs(os.path.dirname(target_file))
            shutil.copy2(source_file, target_file)
        # generating input for performance evaluation
        random.shuffle(test_all_file)
        eval_data = test_all_file[:num_eval]
        eval_input = []
        tokenizer = RegexpTokenizer(r'\w+')
        for path_eval_data in eval_data:
            encoding = get_encoding_of_file(path_eval_data)
            with open(path_eval_data, 'r', encoding=encoding) as file:
                data = file.read()
            que = tokenizer.tokenize(data)[:max_que_len]
            eval_input.append(' '.join(que))
        with open(f'Data/{data_name}-test/eval_input.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(eval_input))


def construct_retrieval_database(data_name_list: List[str],
                                 split_method: List[str] = None,
                                 encoder_model_name: str = 'all-MiniLM-L6-v2',
                                 retrival_database_batch_size: int = 256,
                                 chunk_size: int = 1500,
                                 chunk_overlap: int = 100,
                                 ) -> 'langchain.vectorstores.chroma.Chroma':
    """
    Construct a retrieval database from a dataset or multiple datasets
    :param
    `   data_name_list: The name of the datasets. The datasets are placed in f'./Data/{data_name}'
            optional: ['enron-mail', 'chatdoctor', 'wikitext-103']
            If you run pre_process_dataset, you can also use: 'enron-mail-body', 'enron-mail-strip'
            If you run split_dataset, you can also use: f'{data_name}-train' and f'{data_name}-test'
        split_method: The method to split the data. Each dataset should be provided a split method.
                      The len of the list should be 1 or len(data_name_list) or None
            optional: ['single_file', 'by_two_line_breaks', 'recursive_character']
                single_file: each single file in the datasets is built as a chunk. We use for enron mail.
                by_two_line_breaks: the file is split by '\n\n' to chunks. We use for chatdoctor.
                recursive_character: using RecursiveCharacterTextSplitter in langchain.text_splitter.
                                     We use for wikitext
        encoder_model_name: str. The name of encoder. Default is 'all-MiniLM-L6-v2' from sentence_transformers
            optional: 'open-ai', 'bge-large-en-v1.5', 'all-MiniLM-L6-v2', 'e5-base-v2'
        retrival_database_batch_size: The batch size of the retrieval database for querying the retrieval database
        chunk_size: Only split_method == 'recursive_character' is used. The chunk size of the splitter.
        chunk_overlap: Only split_method == 'recursive_character' is used. The overlap of the splitter
    :return
        A dataset with a retrieval database, the type is langchain.vectorstores.chroma.Chroma
    :function
        get_splitter: get the splitter
    :class
        SingleFileSplitter: constructs a splitter object that splits each file as a chunk
        LineBreakTextSplitter: constructs a splitter object that splits the data by '\n\n'
    """

    class SingleFileSplitter(TextSplitter):
        def split_text(self, text: str) -> List[str]:
            return [text]

    class LineBreakTextSplitter(TextSplitter):
        def split_text(self, text: str) -> List[str]:
            return text.split("\n\n")

    def get_splitter(split_method_) -> SingleFileSplitter:
        splitter_ = None
        if split_method_ == 'single_file':
            splitter_ = SingleFileSplitter()
        elif split_method_ == 'by_two_line_breaks':
            splitter_ = LineBreakTextSplitter()
        elif split_method_ == 'recursive_character':
            splitter_ = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter_

    data_store_path = 'Data'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if split_method is None:
        # No split method provided, default method used
        split_method = ['single_file'] * len(data_name_list)
    elif len(split_method) == 1:
        # Only one split method is provided, this method is used for all the datasets
        split_method = split_method * len(data_name_list)
    else:
        assert len(split_method) == len(data_name_list)
    split_texts = []
    for n_data_name, data_name in enumerate(data_name_list):
        documents = []
        # open the files
        data_path = os.path.join(data_store_path, data_name)
        for file_name in find_all_file(data_path):
            # detect the encode method of files:
            encoding = get_encoding_of_file(file_name)
            # load the data
            loader = TextLoader(file_name, encoding=encoding)
            doc = loader.load()
            documents.extend(doc)

        print(f'File number of {data_name}: {len(documents)}')
        # get the splitter
        splitter = get_splitter(split_method[n_data_name])
        # split the texts
        split_texts += splitter.split_documents(documents)
    embed_model = get_embed_model(encoder_model_name, device, retrival_database_batch_size)
    retrieval_name = '_'.join(data_name_list)
    if len(data_name_list) != 1:
        retrieval_name = 'mix_' + retrieval_name
    vector_store_path = f"./RetrievalBase/{retrieval_name}/{encoder_model_name}"
    print(f'generating chroma database of {retrieval_name} using {encoder_model_name}')
    retrieval_database = Chroma.from_documents(documents=split_texts,
                                               embedding=embed_model,
                                               persist_directory=vector_store_path)
    return retrieval_database


def load_retrieval_database_from_address(store_path: str,
                                         encoder_model_name: str = 'all-MiniLM-L6-v2',
                                         retrival_database_batch_size: int = 512
                                         ) -> 'langchain.vectorstores.chroma.Chroma':
    """
    Load pre-built retrieval database
    :param
        store_path: str. The address of the pre-built retrieval database
        encoder_model_name: str. The name of encoder. Default is 'all-MiniLM-L6-v2' from sentence_transformers
            optional: 'open-ai', 'bge-large-en-v1.5', 'all-MiniLM-L6-v2'
        retrival_database_batch_size: The batch size of the retrieval database for querying the retrieval database
    :return
        A dataset with a retrieval database, the type is langchain.vectorstores.chroma.Chroma
    :important note
        The retrieval database must match the encoder model!
        if encode the database by 'all-MiniLM-L6-v2', can not load the database by 'bge-large-en-v1.5'
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = get_embed_model(encoder_model_name, device, retrival_database_batch_size)
    retrieval_database = Chroma(
        embedding_function=embed_model,
        persist_directory=store_path
    )
    return retrieval_database


def load_retrieval_database_from_parameter(data_name_list: List[str],
                                           encoder_model_name: str = 'all-MiniLM-L6-v2',
                                           retrival_database_batch_size: int = 512
                                           ) -> 'langchain.vectorstores.chroma.Chroma':
    """
    Load the database by some parameters, in this function, it is clearer
    :param
        data_name_list:The name of the datasets. The datasets are placed in f'./Data/{data_name}'
            optional: ['enron-mail', 'chatdoctor', 'wikitext-103']
        encoder_model_name: str. The name of encoder. Default is 'all-MiniLM-L6-v2' from sentence_transformers
            optional: 'open-ai', 'bge-large-en-v1.5', 'all-MiniLM-L6-v2'
        retrival_database_batch_size: The batch size of the retrieval database for querying the retrieval database
    :return:
        A dataset with a retrieval database, the type is langchain.vectorstores.chroma.Chroma
    :important note
        The retrieval database must match the encoder model!
        if encode the database by 'all-MiniLM-L6-v2', can not load the database by 'bge-large-en-v1.5'
    """
    database_store_path = 'RetrievalBase'
    retrieval_name = '_'.join(data_name_list)
    if len(data_name_list) != 1:
        retrieval_name = 'mix_' + retrieval_name
    store_path = f"./{database_store_path}/{retrieval_name}/{encoder_model_name}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = get_embed_model(encoder_model_name, device, retrival_database_batch_size)
    retrieval_database = Chroma(
        embedding_function=embed_model,
        persist_directory=store_path
    )
    return retrieval_database


if __name__ == '__main__':
    """
    You can run following example code to do your experiments, or directly parse parameters
    
    # Step 1: preprocessing the data
    # We have provided the processed data, so you don't need to execute this code.
    # pre_process_dataset('chatdoctor200k')           # num of dataset is 207408
    # run this code to edit the enron-mail data
    pre_process_dataset('enron-mail', 'body')         # num of dataset is 515437
    pre_process_dataset('enron-mail', 'strip')        # num of dataset is 517401
    # split the dataset to train and test if needed
    split_dataset('enron-mail')
    split_dataset('chatdoctor')
    
    # Step 2: Building the vector database
    construct_retrieval_database(['chatdoctor'], ['by_two_line_breaks'], 'bge-large-en-v1.5')    # building chatdoctor
    construct_retrieval_database(['enron-mail'], ['by_two_line_breaks'], 'bge-large-en-v1.5')    # building enron-mail
    construct_retrieval_database(['enron-mail', 'wikitext-103'], ['single_file', 'recursive_character'],
                                 'bge-large-en-v1.5')    # build mix dataset
    construct_retrieval_database(['chatdoctor-train'], ['by_two_line_breaks'], 'bge-large-en-v1.5') # building train set
    # For more details, please read the explanatory note of the function construct_retrieval_database
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--encoder_model', type=str)
    parser.add_argument('--flag_mix', type=bool, default=False)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    encoder_model = args.encoder_model
    flag_mix = args.flag_mix

    if dataset_name.find('body') != -1 and not os.path.exists('Data/enron-mail-body'):
        pre_process_dataset('enron-mail', 'body')
    if dataset_name.find('strip') != -1 and not os.path.exists('Data/enron-mail-strip'):
        pre_process_dataset('enron-mail', 'strip')
    if dataset_name.find('train') != -1 and not os.path.exists(f'Data/{dataset_name}'):
        split_dataset(dataset_name[:-6])
    if dataset_name.find('test') != -1 and not os.path.exists(f'Data/{dataset_name}'):
        split_dataset(dataset_name[:-5])
    if flag_mix is True:
        if dataset_name.find('enron-mail') != -1:
            construct_retrieval_database([dataset_name, 'wikitext-103'],
                                         ['single_file', 'recursive_character'],
                                         encoder_model)
        elif dataset_name.find('chatdoctor') != -1:
            construct_retrieval_database([dataset_name, 'wikitext-103'],
                                         ['by_two_line_breaks', 'recursive_character'],
                                         encoder_model)
    else:
        if dataset_name.find('enron-mail') != -1:
            construct_retrieval_database([dataset_name], ['single_file'], encoder_model)
        elif dataset_name.find('chatdoctor') != -1:
            construct_retrieval_database([dataset_name], ['by_two_line_breaks'], encoder_model)
