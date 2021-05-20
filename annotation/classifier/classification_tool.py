import json
import argparse
import tkinter as tk
from tkinter import ttk
import os
from shutil import copyfile, move
from PIL import ImageTk, Image
from utils.general_utlis import *
from tkinter import messagebox
import operator

"""
Implementation based on https://github.com/Nestak2/image-sorter2
"""


class ClassificationGUI:
    """
    GUI for iFind1 image sorting. This draws the GUI and handles all the events.
    Useful, for sorting views into sub views or for removing outliers from the data.
    """

    def __init__(self, master="", whole_path_list="", anno_output_folder="", dataset="scannet"):
        """
        Initialise GUI
        :param master: The parent window
        :param labels: A list of labels that are associated with the images
        :param whole_path_list: A list of file whole_path_list to images
        :return:
        """

        self.dataset = dataset  # m3d; scannet; nyu
        self.whole_path_list = whole_path_list

        self.anno_output_folder = anno_output_folder

        self.get_progress(start_gui=True)

        self.master = master

        frame = tk.Frame(master)
        frame.grid()

        self.labels = ["mirror", "negative"]
        self.n_labels = 2

        self.image_panel = tk.Label(frame)
        self.set_image(whole_path_list[self.index])

        self.buttons = []
        self.buttons.append(
            tk.Button(frame, text="mirror", height=2, fg='blue', command=lambda l="mirror": self.vote(l))
            .grid(row=0, column=0, sticky='we'))
        self.buttons.append(
            tk.Button(frame, text="negative", height=2, fg='blue', command=lambda l="negative": self.vote(l)).grid(
                row=0, column=1, sticky='we'))
        self.buttons.append(
            tk.Button(frame, text="prev im", height=2, fg="green", command=lambda l="": self.move_prev_image()).grid(
                row=1, column=0, sticky='we'))
        self.buttons.append(
            tk.Button(frame, text="next im", height=2, fg='green', command=lambda l="": self.move_next_image()).grid(
                row=1, column=1, sticky='we'))

        self.mirror_count_label = tk.Label(frame, text="Mirror count: {}".format(len(self.positive_list)))
        self.mirror_count_label.grid(row=2, column=self.n_labels, sticky='we')

        self.mirror_tag_label = tk.Label(frame, text="Mirror tag: {}".format(self.get_tag()))
        self.mirror_tag_label.grid(row=1, column=self.n_labels, sticky='we')

        progress_string = "{}/{}".format(len(self.annotated_paths), len(self.whole_path_list))
        self.progress_label = tk.Label(frame, text=progress_string)
        self.progress_label.grid(row=0, column=self.n_labels, sticky='we')

        self.text_file_name = tk.StringVar()
        self.sorting_label = tk.Entry(root, state='readonly', readonlybackground='white', fg='black',
                                      textvariable=self.text_file_name)
        self.text_file_name.set("index {} in folder: {}".format(self.index, self.whole_path_list[self.index]))
        tk.Label(frame, text="Go to pic (0 ~ {}):".format(len(self.whole_path_list) - 1)).grid(row=2, column=0)

        self.return_ = tk.IntVar()  # return_-> self.index
        self.return_entry = tk.Entry(frame, width=6, textvariable=self.return_)
        self.return_entry.grid(row=2, column=1, sticky='we')
        master.bind('<Return>', self.num_pic_type)

        self.sorting_label.grid(row=4, column=0, sticky='we', columnspan=self.n_labels + 1)
        self.image_panel.grid(row=3, column=0, sticky='we', columnspan=self.n_labels + 1)
        master.bind("q", self.press_key_event)  # mirror
        master.bind("w", self.press_key_event)  # negative
        master.bind('<Left>', self.press_key_event)
        master.bind('<Right>', self.press_key_event)

    def get_tag(self):
        current_tag = "N/A"
        current_path = self.whole_path_list[self.index]
        if current_path in self.annotated_paths:
            if current_path in self.positive_list:
                current_tag = "mirror"
            elif current_path in self.negative_list:
                current_tag = "negative"
        return current_tag

    def num_pic_type(self, event):
        self.index = self.return_.get()
        progress_string = "{}/{}".format(len(self.annotated_paths), len(self.whole_path_list))
        self.progress_label.configure(text=progress_string)
        self.text_file_name.set("index {} in folder: {}".format(self.index, self.whole_path_list[self.index]))
        self.mirror_tag_label.configure(text="Mirror tag: {}".format(self.get_tag()))
        self.set_image(self.whole_path_list[self.index])

    def move_prev_image(self):
        """
        Displays the prev image in the paths list AFTER BUTTON CLICK,
        doesn't update the progress display
        """
        if self.index == 0:
            return
        self.index -= 1
        progress_string = "{}/{}".format(len(self.annotated_paths), len(self.whole_path_list))
        self.progress_label.configure(text=progress_string)
        self.text_file_name.set("index {} in folder: {}".format(self.index, self.whole_path_list[self.index]))
        self.mirror_tag_label.configure(text="Mirror tag: {}".format(self.get_tag()))

        if self.index < len(self.whole_path_list):
            self.set_image(self.whole_path_list[self.index])  # change path to be out of df
        else:
            self.master.quit()

    def move_next_image(self):
        """
        Displays the next image in the paths list AFTER BUTTON CLICK,
        doesn't update the progress display
        """
        if self.index == (len(self.whole_path_list) - 1):
            return
        if self.get_tag() == "N/A":
            messagebox.showerror("error", "please label this sample first!")
            return

        self.index += 1
        progress_string = "{}/{}".format(len(self.annotated_paths), len(self.whole_path_list))
        self.progress_label.configure(text=progress_string)

        self.text_file_name.set("index {} in folder: {}".format(self.index, self.whole_path_list[self.index]))
        self.mirror_tag_label.configure(text="Mirror tag: {}".format(self.get_tag()))

        if self.index < len(self.whole_path_list):
            self.set_image(self.whole_path_list[self.index])
        else:
            self.master.quit()

    def press_key_event(self, event):

        if event.keysym == "q":
            self.vote("mirror")
        elif event.keysym == "w":
            self.vote("negative")
        elif event.keysym == "Left":
            self.move_prev_image()
        elif event.keysym == "Right":
            self.move_next_image()

    def vote(self, label):  # TODO if have voeted then change the txt
        """
        Processes a vote for a label: Initiates the file copying and shows the next image
        :param label: The label that the user voted for
        """
        current_path = self.whole_path_list[self.index]
        if current_path in self.annotated_paths:
            if current_path in self.positive_list:
                self.positive_list.remove(current_path)
            elif current_path in self.negative_list:
                self.negative_list.remove(current_path)

        if label == "mirror":
            self.positive_list.append(current_path)
            self.annotated_paths.append(current_path)
        else:
            self.negative_list.append(current_path)
            self.annotated_paths.append(current_path)

        self.mirror_tag_label.configure(text="Mirror count: {}".format(len(self.positive_list)))
        self.save_progress()
        self.show_next_image()

    def show_next_image(self):
        """
        Displays the next image in the whole_path_list list and updates the progress display
        """
        self.index += 1
        progress_string = "{}/{}".format(len(self.annotated_paths), len(self.whole_path_list))
        self.progress_label.configure(text=progress_string)
        self.mirror_count_label.configure(text=len(self.positive_list))
        self.mirror_tag_label.configure(text="Mirror tag: {}".format(self.get_tag()))
        self.text_file_name.set("index {} in folder: {}".format(self.index, self.whole_path_list[self.index]))
        if self.index < len(self.whole_path_list):
            self.set_image(self.whole_path_list[self.index])  # TODO
        else:
            self.master.quit()

    @staticmethod
    def _load_image(path):  # TODO to inhereate & delete
        """
        Loads and resizes an image from a given path using the Pillow library
        :param path: Path to image
        :return: Resized or original image 
        """
        image = Image.open(path)
        max_height = 500
        img = image
        s = img.size
        ratio = max_height / s[1]
        image = img.resize((int(s[0] * ratio), int(s[1] * ratio)), Image.ANTIALIAS)
        return image

    def set_image(self, path):  # TODO to inhereate & delete
        """
        Helper function which sets a new image in the image view
        :param path: path to that image
        """
        image = self._load_image(path)
        self.image_raw = image
        self.image = ImageTk.PhotoImage(image)
        self.image_panel.configure(image=self.image)

    def save_progress(self):
        """Save annotation progress"""
        anotation_progress_save_folder = os.path.join(self.anno_output_folder, "classification_progress")
        neg_txt_path = os.path.join(anotation_progress_save_folder, "negative_list.txt")
        pos_txt_path = os.path.join(anotation_progress_save_folder, "positive_list.txt")
        annotated_txt_path = os.path.join(anotation_progress_save_folder, "annotated_list.txt")
        save_txt(neg_txt_path, self.negative_list)
        save_txt(pos_txt_path, self.positive_list)
        save_txt(annotated_txt_path, self.annotated_paths)
        self.get_progress()

    def get_progress(self, start_gui=False):
        """Get annotation progress"""
        anotation_progress_save_folder = os.path.join(self.anno_output_folder, "classification_progress")
        os.makedirs(anotation_progress_save_folder, exist_ok=True)

        error_txt = os.path.join(anotation_progress_save_folder, "negative_list.txt")
        correct_txt = os.path.join(anotation_progress_save_folder, "positive_list.txt")
        annotated_txt_path = os.path.join(anotation_progress_save_folder, "annotated_list.txt")

        if os.path.exists(error_txt):
            self.negative_list = read_txt(error_txt)
        else:
            self.negative_list = []

        if os.path.exists(correct_txt):
            self.positive_list = read_txt(correct_txt)
        else:
            self.positive_list = []

        if os.path.exists(annotated_txt_path):
            self.annotated_paths = read_txt(annotated_txt_path)
        else:
            self.annotated_paths = []

        if start_gui:
            self.index = len(self.annotated_paths) - 1
            if self.index == -1:
                self.index = 0

        self.path_to_annotate = list_diff(self.whole_path_list, self.annotated_paths)
        return self.annotated_paths, self.path_to_annotate, self.negative_list, self.positive_list


def save_for_cavt(pos_list, cvat_folder, dataset_name):
    os.makedirs(cvat_folder, exist_ok=True)
    for pos_sample_path in pos_list:
        if dataset_name == "scannet":
            scannet_sample_name = "{}_{}".format(pos_sample_path.split("/")[-3], pos_sample_path.split("/")[-1])
            save_path = os.path.join(cvat_folder, scannet_sample_name)
            shutil.copy(pos_sample_path, save_path)
        else:
            save_path = os.path.join(cvat_folder, pos_sample_path.split("/")[-1])
            shutil.copy(pos_sample_path, save_path)
    print("smaples for CVAT annotation are copied to : {}".format(cvat_folder))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_root', help='Input folder where the *tif images should be', default="")
    parser.add_argument('-j', '--json_file_path', help='Json file consist of input file names', default="")
    parser.add_argument('-o', '--anno_output_folder', help='annotation result output folder', default="")
    parser.add_argument('-s', '--stage',
                        help='(1) annotation tool (2) generate retrain list (3) generate mirror package for CVAT '
                             'annotation',
                        default="1")
    parser.add_argument('-ot', '--train_info_output_folder',
                        help='stage 2 parameter : training information output folder', default="")
    parser.add_argument('-p', '--pos_txt', help='stage 2 parameter : previous all positive path list txt', default="")
    parser.add_argument('-n', '--neg_txt', help='stage 2 parameter : previous all negative path list txt', default="")
    parser.add_argument('-ocvat', '--cvat_output_folder', help='stage 2 parameter : output folder for cvat', default="")
    parser.add_argument('-d', '--dataset_name', help='stage 2 parameter : dataset name', default="scannet")
    args = parser.parse_args()

    file_score_list = sorted(read_json(args.json_file_path).items(), key=operator.itemgetter(1), reverse=True)
    file_path_abv = [i[0] for i in file_score_list]
    paths = [os.path.join(args.data_root, file_name) for file_name in file_path_abv]

    if args.stage == "1":

        root = tk.Tk()
        root.title("Mirror Classification Tool")
        root.protocol('WM_DELETE_WINDOW')
        app = ClassificationGUI(master=root, whole_path_list=paths, anno_output_folder=args.anno_output_folder,
                                dataset=args.dataset_name)
        root.mainloop()
    elif args.stage == "2":

        anotation_progress_save_folder = os.path.join(args.anno_output_folder, "classification_progress")
        anno_neg_txt_path = os.path.join(anotation_progress_save_folder, "negative_list.txt")
        anno_pos_txt_path = os.path.join(anotation_progress_save_folder, "positive_list.txt")
        pos_anno = read_txt(anno_pos_txt_path)
        neg_anno = read_txt(anno_neg_txt_path)
        to_anno = list(set(paths) - set(neg_anno) - set(pos_anno))
        pos_new = read_txt(args.pos_txt) + pos_anno
        neg_new = read_txt(args.neg_txt) + neg_anno
        os.makedirs(args.train_info_output_folder, exist_ok=True)
        new_neg_txt_savepath = os.path.join(args.train_info_output_folder, "train_negative.txt")
        new_pos_txt_savepath = os.path.join(args.train_info_output_folder, "train_positive.txt")
        to_anno_txt_savepath = os.path.join(args.train_info_output_folder, "to_anno_positive.txt")
        save_txt(new_neg_txt_savepath, neg_new)
        save_txt(new_pos_txt_savepath, pos_new)
        save_txt(to_anno_txt_savepath, to_anno)
        save_for_cavt(pos_anno, args.cvat_output_folder, args.dataset_name)
