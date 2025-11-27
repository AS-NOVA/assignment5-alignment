from typing import Iterator, Any
import json

from itertools import islice
def get_limited_iter(iter:Iterator, stop:int)->Iterator:
    """
    获取有限长度的迭代器。
    Args:
        iter (Iterator): 输入的迭代器。
        stop (int): 迭代器的最大长度。
    Returns:
        output(Iterator): 限制长度后的迭代器。
    """
    return islice(iter,stop)

def read_jsonl(path:str)->Iterator[dict]:
    """
    逐行读取一个jsonl文件，返回一迭代器，其可以跳过jsonl文件中的空行，并将每行json解析为一个字典。
    Args:
        path (str): jsonl文件路径。
    Yields:
        output (dict): 每一行 JSON 解析后的字典对象。
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)




def get_qa_list(qaiter:Iterator)->list[dict[str,Any]]:
    """
    把迭代器里的qa读到列表里
    """
    qas = []
    for qa in qaiter:
        #quick_look_at_qa(qa)
        qas.append(qa)
    return qas



def quick_look_at_qa(qa:dict[str,Any]):
    """
    快速查看一个形如{"question":str,"answer":str}的问答对的内容。
    """
    q = qa["question"]
    a = qa["answer"]
    print("--------question--------")
    print(q)
    print("--------answer--------")
    print(a)
    print()
    

