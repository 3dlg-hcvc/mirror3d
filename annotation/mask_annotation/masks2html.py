import argparse
from tqdm import trange
import os
import bs4

LABEL_COLOR_MAP = {"Complete": "rgb(221, 228, 57)", "Incomplete": "rgb(241, 80, 149)",
                   "Almost-complete": "rgb(108, 96, 74)"}


def generate_html(args_obj):
    """
    Generate html to show masks; all views for one sample is shown in one line;
    """

    os.makedirs(args_obj.output, exist_ok=True)
    img_ids = os.listdir(args_obj.data)
    count = 0
    one_line_mask_info = []
    for img_id in img_ids:
        img_folder_path = os.path.abspath(os.path.join(args_obj.data, img_id))
        masks = os.listdir(img_folder_path)
        if len(masks) == 0:
            continue
        count += (len(masks) - 1) // 2
        with open(os.path.join(img_folder_path, "labels.txt"), "r") as f:
            labels = eval(f.readline().strip())
        for coarse_instance_mask in masks:
            if coarse_instance_mask[-19:] == "coarse_instance.png":
                detailed_instance_mask = coarse_instance_mask.replace("coarse", "detailed")
                coarse_instance_id = coarse_instance_mask[:-20]
                one_line_mask_info.append(
                    (img_id, coarse_instance_id, detailed_instance_mask[:-22],
                     os.path.join(img_folder_path, coarse_instance_mask).replace("/project/3dlg-hcvc/mirrors/www", "/projects/mirrors"),
                     os.path.join(img_folder_path, detailed_instance_mask).replace("/project/3dlg-hcvc/mirrors/www", "/projects/mirrors"),
                     labels[coarse_instance_id + "_coarse_instance"]
                     )
                )

    page_num = count // args_obj.instance_num_per_page + 1

    for page in range(page_num):

        with open(args_obj.html_template_path) as inf:
            txt = inf.read()
            soup = bs4.BeautifulSoup(txt, features="html.parser")

        for i in range(args_obj.instance_num_per_page):
            index = i + (page * args_obj.instance_num_per_page)
            if index >= len(one_line_mask_info):
                break
            one_line = one_line_mask_info[index]

            new_div = soup.new_tag("div")
            new_div['class'] = "one-instance"
            soup.body.append(new_div)

            # Append text to one line in HTML
            id_box = soup.new_tag("div")
            id_box["style"] = "text-align:center;"
            id_text = soup.new_tag("div")
            id_text["class"] = "one-item"
            id_text['style'] = "padding-top: 170px;"
            id_text.string = one_line[0]
            id_box.append(id_text)
            id_label = soup.new_tag("div")
            id_label["style"] = "font-size: 20px; font-weight: bold; color: " + LABEL_COLOR_MAP[one_line[5]]
            id_label.string = "Label: " + one_line[5]
            id_box.append(id_label)
            new_div.append(id_box)

            # Append coarse instance mask overlay to one line in HTML
            coarse_mask_box = soup.new_tag("div")
            coarse_mask_box["style"] = "text-align:center;"
            coarse_mask = soup.new_tag("div")
            coarse_mask["class"] = "one-item"
            coarse_mask.append(soup.new_tag('img', src=one_line[3]))
            coarse_mask_box.append(coarse_mask)
            instance_id = soup.new_tag("span")
            instance_id.string = one_line[1]
            coarse_mask_box.append(instance_id)
            new_div.append(coarse_mask_box)

            # Append detailed instance mask overlay to one line in HTML
            detailed_mask_box = soup.new_tag("div")
            detailed_mask_box["style"] = "text-align:center;"
            detailed_mask = soup.new_tag("div")
            detailed_mask["class"] = "one-item"
            detailed_mask.append(soup.new_tag('img', src=one_line[4]))
            detailed_mask_box.append(detailed_mask)
            instance_id = soup.new_tag("span")
            instance_id.string = one_line[2]
            detailed_mask_box.append(instance_id)
            new_div.append(detailed_mask_box)

        html_path = os.path.join(args_obj.output, "{}.html".format(page))

        with open(html_path, "w") as f:
            f.write(str(soup))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='The data main folder')
    parser.add_argument('-o', '--output', help='The output folder', default="output")
    parser.add_argument('-i', '--instance_num_per_page', help='The number of mask instances per html page',
                        default=100, type=int)
    parser.add_argument('-t', '--html_template_path', help='The html template')
    args = parser.parse_args()
    generate_html(args)
