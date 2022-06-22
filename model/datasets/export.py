import json


def export_classes(characters, dest):
    info_kinds = ["work", "name"]

    char_infos = [
        {info_kinds[i]: char.split("::::::")[i] for i in range(2)}
        for char in characters
    ]

    with open(dest, "w", encoding="utf-8") as json_file:
        json.dump(char_infos, json_file, ensure_ascii=False)
