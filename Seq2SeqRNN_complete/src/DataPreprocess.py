# -*- encoding: utf-8 -*-

import json

dir_path = "../data/film/"  # 由于每个领域知识图谱是分开的，所以需要对每个领域单独处理
file_names = ("train.json", "dev.json", "test.json")


def preprocess(dir_path: str, file_name: str):
    with open(dir_path + file_name, 'r', encoding='utf-8') as in_file:
        with open(dir_path + "processed_" + file_name, 'w', encoding='utf-8') as out_file:
            in_file = json.load(in_file)
            res = []
            for session in in_file:
                # last_entities = [session["name"]]
                last_message = ''
                session = session["messages"]
                for line in session:
                    now_message = line["message"]
                    if "attrs" in line:
                        for knowledge in line['attrs']:
                            last_message += knowledge["name"] + knowledge["attrname"] + knowledge["attrvalue"]
                    if last_message:
                        res.append((last_message, now_message))
                        # res.append((line["message"], line["message"]))  # 验证模型
                    last_message = now_message
            json.dump(res, out_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    for file_name in file_names:
        preprocess(dir_path, file_name)
