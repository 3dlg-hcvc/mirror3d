import os

def save_txt(save_path, data):
    with open(save_path, "w") as file:
        for info in data:
            file.write(str(info))
            file.write("\n")
    print("txt saved to : ", save_path, len(data))

def read_txt(txt_path):
    with open(txt_path, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]