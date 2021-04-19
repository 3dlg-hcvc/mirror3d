from PIL import Image
from tqdm import tqdm
import os
import argparse
import multiprocessing

DETAILED_MASK_PATH = "precise"
COARSE_MASK_PATH = "coarse"
INSTANCE_MASK_PATH = "instance_mask"
SEMANTIC_MASK_PATH = "semantic_mask"
LABEL_MAP = {(108, 96, 74): "Almost-complete", (221, 228, 57): "Complete", (241, 80, 149): "Incomplete"}
FINAL_MASK_FILENAME_SUFFIXES = ("_coarse_instance.png", "_detailed_instance.png")


def generate_overlay_masks_and_check(args_obj, raw_filenames):
    coarse_mask_path = os.path.join(args_obj.masks_folder, COARSE_MASK_PATH)
    detailed_mask_path = os.path.join(args_obj.masks_folder, DETAILED_MASK_PATH)
    semantic_mask_paths = (os.path.join(coarse_mask_path, SEMANTIC_MASK_PATH),
                           os.path.join(detailed_mask_path, SEMANTIC_MASK_PATH))
    label_error_list = []
    for file in tqdm(raw_filenames):
        path = os.path.join(args_obj.output, file[:-4])
        raw = Image.open(os.path.join(args_obj.raw_folder, file))
        coarse_instance_mask_path = os.path.join(coarse_mask_path, INSTANCE_MASK_PATH, file)
        detailed_instance_mask_path = os.path.join(detailed_mask_path, INSTANCE_MASK_PATH, file)
        if not os.path.exists(coarse_instance_mask_path) or not os.path.exists(detailed_instance_mask_path):
            continue
        os.makedirs(path, exist_ok=True)
        mask_list = (Image.open(coarse_instance_mask_path),
                     Image.open(detailed_instance_mask_path))
        width, height = raw.size
        colors = mask_list[0].getcolors()
        labels = dict()

        for i in range(len(mask_list)):
            semantic_mask = Image.open(os.path.join(semantic_mask_paths[i], file))
            color_dic = dict()

            for color in colors:
                if color[1] != (0, 0, 0):
                    color_dic[color[1]] = []

            for w in range(width):
                for h in range(height):
                    tmp_pixel = mask_list[i].getpixel((w, h))
                    if tmp_pixel != (0, 0, 0):
                        color_dic[tmp_pixel].append((w, h))

            for key, item in color_dic.items():
                tmp_img = Image.new("RGBA", (width, height))
                for pixel in item:
                    tmp_img.putpixel(pixel, (key[0], key[1], key[2], args_obj.alpha))
                final = raw.copy()
                final.paste(tmp_img, (0, 0), mask=tmp_img)
                mask_id = str(key[2]) + "_" + str(key[1]) + "_" + str(key[0]) + FINAL_MASK_FILENAME_SUFFIXES[i]
                final.save(os.path.join(path,  mask_id))
                labels[mask_id[:-4]] = LABEL_MAP[semantic_mask.getpixel(item[len(item) // 2])]

        with open(os.path.join(path, "labels.txt"), "w") as f:
            f.write(str(labels))

        for key, value in labels.items():
            if key[-15:] == "coarse_instance":
                if labels[key.replace("coarse", "detailed")] != value:
                    label_error_list.append(file[:-4] + "_".join(key.split("_")[0:2]))

    return label_error_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_folder', help='The Raw folder')
    parser.add_argument('--masks_folder',  help='The Mask folder that consists of coarse and detailed masks')
    parser.add_argument('--output', help='The output folder', default="output")
    parser.add_argument('--alpha', help='The alpha value of overlays (0-255)', default=165, type=int)
    args = parser.parse_args()

    total_raw_filenames = os.listdir(args.raw_folder)
    coarse_mask_path = os.path.join(args.masks_folder, COARSE_MASK_PATH)
    detailed_mask_path = os.path.join(args.masks_folder, DETAILED_MASK_PATH)
    # Check consistency
    len_raw = len(total_raw_filenames)
    len_coarse_instance = len(os.listdir(os.path.join(coarse_mask_path, INSTANCE_MASK_PATH)))
    len_detailed_instance = len(os.listdir(os.path.join(detailed_mask_path, INSTANCE_MASK_PATH)))
    len_coarse_semantic = len(os.listdir(os.path.join(coarse_mask_path, SEMANTIC_MASK_PATH)))
    len_detailed_semantic = len(os.listdir(os.path.join(detailed_mask_path, SEMANTIC_MASK_PATH)))
    if len_raw * 4 != len_coarse_instance + len_detailed_instance + len_coarse_semantic + len_detailed_semantic:
        print("\nWarning: Inconsistent number of files\n")
        print("Raw image: " + str(len_raw))
        print("Instance mask(coarse): " + str(len_coarse_instance))
        print("Instance mask(detailed): " + str(len_detailed_instance))
        print("Semantic mask(coarse): " + str(len_coarse_semantic))
        print("Semantic mask(detailed): " + str(len_detailed_semantic) + "\n")
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    results = []
    print("CPU Logical Cores:", cpu_count)
    for i in range(0, len_raw, len_raw // cpu_count):
        results.append(pool.apply_async(generate_overlay_masks_and_check, args=(args, total_raw_filenames[i:i + len_raw // cpu_count])))
    label_error_list = []
    for result in results:
        label_error_list += result.get()
    print("\nDone.")

    if len(label_error_list) != 0:
        log_path = os.path.abspath(os.path.join(args.output, "label_error_ids.txt"))
        print("\nWarning: " + str(
            len(label_error_list)) + " masks have inconsistent labels, ids store at " + log_path + "\n")
        with open(log_path, "w") as f:
            for error_id in label_error_list:
                f.write(error_id + "\n")
