import argparse
import sys
from utils.general_utlis import *


def transform_table(json_file_path):
    dataset = json_file_path.split("_")[1]
    tag_info = read_json(json_file_path)

    sensor_methods = ["*", "\mnet"]
    rgbd_order = ["saic~\\cite{senushkin2020decoder}", "saic~\\cite{senushkin2020decoder} + \mnet",
                  "PlaneRCNN~\\cite{liu2019planercnn}", "\mnet"]
    rgb_order = ["BTS~\cite{lee2019big}", "BTS~\cite{lee2019big} + \mnet", "VNL~\cite{yin2019enforcing}",
                 "VNL~\cite{yin2019enforcing} + \mnet"]

    sensor_lines = []
    rgbd_lines = []
    rgb_lines = []
    # TODO generate main table and sup table
    for item in tag_info:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reformat latex table')
    parser.add_argument('-i', '--input', default="output/logFile_info.json", help='json info file to transform')
    args = parser.parse_args()
