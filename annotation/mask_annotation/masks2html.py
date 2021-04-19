import argparse
from tqdm import trange
import os
import bs4

LABEL_COLOR_MAP = {"Complete": "rgb(221, 228, 57)", "Incomplete": "rgb(241, 80, 149)",
                   "Almost-complete": "rgb(108, 96, 74)"}

html = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
</head>

<body>
    <div>
        <table style="width: 85%; margin: 0 auto; text-align: center; border-spacing: 35px;" >
            <tr style="font-size: 28px; font-weight: bold; box-shadow: 0 2px 4px rgb(0 0 0 / 12%), 0 0 6px rgb(0 0 0 / 4%);">
                <td style="width:20%;"><p>ID</p></td>
                <td style="width:40%;"><p>Coarse Instance Mask Overlay</p></td>
                <td style="width:40%;"><p>Detailed Instance Mask Overlay</p></td>
            </tr>
        </table>
    </div>
</body>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
<script type="text/javascript">
    // Save scroll position
    $(document).ready(function () {
        if (localStorage.getItem("my_app_name_here-quote-scroll") != null) {
            $(window).scrollTop(localStorage.getItem("my_app_name_here-quote-scroll"));
        }
        $(window).on("scroll", function () {
            localStorage.setItem("my_app_name_here-quote-scroll", $(window).scrollTop());
        });
    });
</script>
</html>
'''


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

        soup = bs4.BeautifulSoup(html, features="html.parser")

        for i in range(args_obj.instance_num_per_page):
            index = i + (page * args_obj.instance_num_per_page)
            if index >= len(one_line_mask_info):
                break
            one_line = one_line_mask_info[index]

            new_div = soup.new_tag("tr")
            new_div["style"] = "box-shadow: 0 2px 12px 0 rgb(0 0 0 / 10%); margin-bottom: 10px;"
            soup.body.div.table.append(new_div)

            # Append text to one line in HTML
            id_box = soup.new_tag("td")
            id_box["style"] = "width: 20%; padding: 30px 0"
            id_text = soup.new_tag("p")
            id_text.string = one_line[0]
            id_box.append(id_text)
            id_label = soup.new_tag("p")
            id_label["style"] = "font-weight: bold; color: " + LABEL_COLOR_MAP[one_line[5]]
            id_label.string = "Label: " + one_line[5]
            id_box.append(id_label)
            new_div.append(id_box)

            # Append coarse instance mask overlay to one line in HTML
            coarse_mask_box = soup.new_tag("td")
            coarse_mask_box["style"] = "width: 40%; padding: 30px 0"
            coarse_mask_img = soup.new_tag('img', src=one_line[3])
            coarse_mask_img["style"] = "max-width: 600px; max-height: 600px; width: 100%; height: auto; object-fit: contain;"
            coarse_mask_box.append(coarse_mask_img)
            instance_id = soup.new_tag("p")
            instance_id["style"] = "font-size: 15px;"
            instance_id.string = one_line[1]
            coarse_mask_box.append(instance_id)
            new_div.append(coarse_mask_box)

            # Append detailed instance mask overlay to one line in HTML
            detailed_mask_box = soup.new_tag("td")
            detailed_mask_box["style"] = "width: 40%; padding: 30px 0"
            detailed_mask_img = soup.new_tag('img', src=one_line[4])
            detailed_mask_img["style"] = "max-width: 600px; max-height: 600px; width: 100%; height: auto; object-fit: contain;"
            detailed_mask_box.append(detailed_mask_img)
            instance_id = soup.new_tag("p")
            instance_id["style"] = "margin-top: 5px; font-size: 16px;"
            instance_id.string = one_line[2]
            detailed_mask_box.append(instance_id)
            new_div.append(detailed_mask_box)

        html_path = os.path.join(args_obj.output, "{}.html".format(page))

        with open(html_path, "w") as f:
            f.write(str(soup))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='The data main folder')
    parser.add_argument('--output', help='The output folder', default="output")
    parser.add_argument('--instance_num_per_page', help='The number of mask instances per html page',
                        default=100, type=int)
    args = parser.parse_args()
    generate_html(args)
