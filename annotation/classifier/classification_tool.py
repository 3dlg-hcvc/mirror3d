import json
import argparse
import tkinter as tk
from tkinter import ttk
import os
from shutil import copyfile, move
from PIL import ImageTk, Image
from utils.general_utlis import *
from tkinter import messagebox


class Classification_GUI:
    """
    GUI for iFind1 image sorting. This draws the GUI and handles all the events.
    Useful, for sorting views into sub views or for removing outliers from the data.
    """

    def __init__(self, master, labels, whole_path_list, anno_output_folder, dataset="scannet"):
        """
        Initialise GUI
        :param master: The parent window
        :param labels: A list of labels that are associated with the images
        :param whole_path_list: A list of file whole_path_list to images
        :return:
        """

        self.dataset = dataset # m3d; scannet; nyu
        self.whole_path_list = whole_path_list


        self.anno_output_folder = anno_output_folder

        self.get_progress(start_gui=True)

        self.master = master

        frame = tk.Frame(master)
        frame.grid()
        
        
        self.labels = labels
        self.n_labels = len(labels)

        self.image_panel = tk.Label(frame)
        self.set_image(whole_path_list[self.index])

        self.buttons = []
        # for class_index, label in enumerate(labels):
        #     self.buttons.append(tk.Button(frame, text=label, height=2, fg='blue', command=lambda l=label: self.vote(l)).grid(row=0, column=class_index, sticky='we'))
        self.buttons.append(tk.Button(frame, text="mirror", height=2, fg='blue', command=lambda l="mirror": self.vote(l)).grid(row=0, column=0, sticky='we'))
        self.buttons.append(tk.Button(frame, text="negative", height=2, fg='blue', command=lambda l="negative": self.vote(l)).grid(row=0, column=1, sticky='we'))
        self.buttons.append(tk.Button(frame, text="prev im", height=2, fg="green", command=lambda l="": self.move_prev_image()).grid(row=1, column=0, sticky='we'))
        self.buttons.append(tk.Button(frame, text="next im", height=2, fg='green', command=lambda l="": self.move_next_image()).grid(row=1, column=1, sticky='we'))

        self.mirror_count_label = tk.Label(frame, text="Mirror count: {}".format(len(self.positive_list)))
        self.mirror_count_label.grid(row=2, column=self.n_labels, sticky='we')

        self.mirror_tag_label = tk.Label(frame, text="Mirror tag: {}".format(self.get_tag()))
        self.mirror_tag_label.grid(row=1, column=self.n_labels, sticky='we')

        progress_string = "{}/{}".format(len(self.annotated_paths), len(self.whole_path_list))
        self.progress_label = tk.Label(frame, text=progress_string)
        self.progress_label.grid(row=0, column=self.n_labels, sticky='we') 

        self.text_file_name = tk.StringVar()
        self.sorting_label = tk.Entry(root, state='readonly', readonlybackground='white', fg='black', textvariable=self.text_file_name)
        self.text_file_name.set("index {} in folder: {}".format(self.index, self.whole_path_list[self.index]))
        tk.Label(frame, text="Go to pic (0 ~ {}):".format(len(self.whole_path_list)-1)).grid(row=2, column=0)

        self.return_ = tk.IntVar() # return_-> self.index
        self.return_entry = tk.Entry(frame, width=6, textvariable=self.return_)
        self.return_entry.grid(row=2, column=1, sticky='we')
        master.bind('<Return>', self.num_pic_type)
        
        self.sorting_label.grid(row=4, column=0, sticky='we',columnspan=self.n_labels+1)
        self.image_panel.grid(row=3, column=0, sticky='we',columnspan=self.n_labels+1)
        master.bind("q", self.press_key_event) # mirror
        master.bind("w", self.press_key_event) # negative
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
            self.set_image(self.whole_path_list[self.index]) # change path to be out of df
        else:
            self.master.quit()

    def move_next_image(self):
        """
        Displays the next image in the paths list AFTER BUTTON CLICK,
        doesn't update the progress display
        """
        if self.index == (len(self.whole_path_list)-1) :
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
        
    def vote(self, label): # TODO if have voeted then change the txt
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
            self.set_image(self.whole_path_list[self.index]) # TODO
        else:
            self.master.quit()

    @staticmethod
    def _load_image(path): # TODO to inhereate & delete
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
        image = img.resize((int(s[0]*ratio), int(s[1]*ratio)), Image.ANTIALIAS)
        return image

    def set_image(self, path): # TODO to inhereate & delete
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
            if self.index == -1 :
                self.index = 0
        
        self.path_to_annotate = list_diff(self.whole_path_list, self.annotated_paths)
        return self.annotated_paths, self.path_to_annotate, self.negative_list, self.positive_list




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_root', help='Input folder where the *tif images should be', default="/local-scratch/jiaqit/exp/data/scannet/scannet_frames_25k")
    parser.add_argument('-j', '--json_file_path', help='Json file consist of input file names', default="/local-scratch/wla172/scannet/extension_annot/epochs/epoch2/imgPath_score_2021-03-03-12-24-35.json")
    parser.add_argument('-e', '--exclusion', help='Exclusion file', required=False)
    parser.add_argument('-o', '--output_file_path', help='Output file name', default="/local-scratch/jiaqit/exp/output/random_test.txt")
    parser.add_argument('-i', '--train_val_info_folder', help='Train val info folder', default="/local-scratch/wla172/scannet/extension_annot/labels")
    args = parser.parse_args()


    file_names = [item[0] for item in read_json(args.json_file_path).items()]


    labels = ["mirror", "negative"]
    paths = [os.path.join(args.data_root,file_name) for file_name in file_names]

    root = tk.Tk()
    root.title("Mirror Classification Tool")
    root.protocol('WM_DELETE_WINDOW')
    app = Classification_GUI(root, labels, paths,"/local-scratch/jiaqit/exp/output/waste")
    root.mainloop()