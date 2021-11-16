import json
import py2neo
import time

MAX_DISTANCE = 3

dir_paths = ["../data/travel/"]  # ["../data/film/", "../data/music/", "../data/travel/"]
file_names = ["test.json"]  # ["train.json", "dev.json", "test.json"]
# graph = py2neo.Graph('http://localhost:7474', auth=("neo4j", ""))
# matcher = py2neo.NodeMatcher(graph)
ans = [0] * (MAX_DISTANCE + 2)  # 最后一个表示超过此距离的


def triple2nodes(triple: str) -> [str]:
    nodes = triple.split('-')
    nodes.pop(1)
    nodes[0] = nodes[0][1: -1]
    nodes[1] = nodes[1][2: -1]
    return nodes


def judge_distance(new_entity: str, old_entities: [str], dis: int) -> bool:
    sql = "MATCH res=(n)-[*1..{}]-(m) where n.name = '{}' return res".format(dis, new_entity.replace("'", r"\'"))
    graph = py2neo.Graph('http://localhost:7474', auth=("neo4j", ""))
    for sql_res in graph.run(sql):
        for triple in sql_res[0]:
            nodes = triple2nodes(str(triple))
            if nodes[0] in old_entities or nodes[1] in old_entities:
                return True
    return False


def count_distance(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as in_file:
        in_file = json.load(in_file)
        for session in in_file:
            last_entities = [session["name"]]
            session = session["messages"]
            for line in session:
                if "attrs" in line:
                    print("上一句实体:", last_entities)
                    for triple in line["attrs"]:
                        updated = False
                        for dis in range(1, MAX_DISTANCE + 1):
                            if judge_distance(triple["name"], last_entities, dis) \
                                    and judge_distance(triple["attrvalue"], last_entities, dis):
                                ans[dis] += 1
                                updated = True
                                break
                        if not updated:
                            ans[MAX_DISTANCE + 1] += 1
                    # 更新上一句的实体
                    last_entities = []
                    for triple in line["attrs"]:
                        last_entities.append(triple["name"])
                        last_entities.append(triple["attrvalue"])
                    last_entities = list(set(last_entities))
            for i in range(len(ans)):
                print("dis:", i, "num:", ans[i])
            # time.sleep(10)


if __name__ == '__main__':
    for dir_path in dir_paths:
        for file_name in file_names:
            count_distance(dir_path + file_name)
        print(dir_path + file_name, ":")
        for i in range(len(ans)):
            print("dis:", i, "num:", ans[i])
