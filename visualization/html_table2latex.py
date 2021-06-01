import bs4
import os
import argparse
import zipfile
from PIL import Image
from shutil import copyfile, rmtree

LATEX_HEADER = '''
\\documentclass{article}
\\usepackage{tabularx}
\\usepackage{graphicx}
\\usepackage{array}
\\usepackage{tikz}
\\usetikzlibrary{tikzmark, calc, fit}
\\renewcommand\\tabularxcolumn[1]{m{#1}}
\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}
\\begin{document}
'''

LATEX_TEMPLATE = '''
\\begin{figure*}
\\setkeys{Gin}{width=\\linewidth}
\\begin{tabularx}{\\textwidth}{Y Y@{\\hspace{1mm}} Y@{\\hspace{0mm}} Y@{\\hspace{1mm}} Y@{\\hspace{1mm}} Y@{\\hspace{2mm}} | Y@{\\hspace{1mm}} Y@{\\hspace{1mm}} Y@{\\hspace{2mm}} | Y@{\\hspace{1mm}} Y@{\\hspace{1mm}} Y@{\\hspace{0mm}} }
 & & & \\multicolumn{3}{c}{\\textcolor{black!60!green}{\\small example-text}} & \\multicolumn{3}{c}{\\textcolor{blue!60!cyan}{\\small example-text}} & \\multicolumn{3}{c}{\\textcolor{cyan}{\\small example-text}}\\\\
& \\small Input Depth (D) &  & \\vspace{4pt}\\tikzmarknode{A1}{\\includegraphics{example-image}} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\tikzmarknode{A2}{\\includegraphics{example-image}} & \\vspace{4pt}\\tikzmarknode{B1}{\\includegraphics{example-image}} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\tikzmarknode{B2}{\\includegraphics{example-image}} & \\vspace{4pt}\\tikzmarknode{C1}{\\includegraphics{example-image}} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\tikzmarknode{C2}{\\includegraphics{example-image}}
\\\\ &
\\vspace{4pt}\\includegraphics{example-image} & \\multicolumn{1}{r}{\\scriptsize example-text} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} \\\\\\cline{2-12} & \\vspace{4pt}\\includegraphics{example-image} & \\multicolumn{1}{r}{\\scriptsize example-text} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} \\\\\\cline{4-12}
& \\small Color (RGB) & \\multicolumn{1}{r}{\\scriptsize example-text} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} \\\\
 & & & \\scriptsize Depth & \\scriptsize Error & \\scriptsize PC & \\scriptsize Depth & \\scriptsize Error & \\scriptsize PC & \\scriptsize Depth & \\scriptsize Error & \\scriptsize PC \\\\
 & & & \\multicolumn{3}{c}{\\small example-text} & \\multicolumn{3}{c}{\\small example-text} & \\multicolumn{3}{c}{\\small example-text}\\\\

\\end{tabularx}
\\begin{tikzpicture}[overlay,remember picture]
\\node[ draw=black!60!green, line width=1pt, fit={(A1)(A2)($(A1.north west)+(+2pt,-0pt)$)($(A2.south east)+(-2pt,-0pt)$)}]{};
\\node[ draw=blue!60!cyan, line width=1pt,fit={(B1)(B2)($(B1.north west)+(+2pt,-0pt)$)($(B2.south east)+(-2pt,-0pt)$)}]{};
\\node[ draw=cyan, line width=1pt,fit={(C1)(C2)($(C1.north west)+(+2pt,-0pt)$)($(C2.south east)+(-2pt,-0pt)$)}]{};
\\end{tikzpicture}
\\caption{Example figure with tabular graphics.}
\\label{fig:example}
\\end{figure*}
'''

LATEX_FOOTER = '''
\\end{document}
'''


def resize_and_add_padding(im, desired_size):
    im = im.resize((desired_size[1], desired_size[1]))
    new_im = Image.new("RGB", (desired_size[0], desired_size[1]), "WHITE")
    new_im.paste(im, ((desired_size[0] - desired_size[1]) // 2, 0))
    return new_im


def extract_table_tags_from_html_and_merge(path):
    table_list = []
    if os.path.isdir(path):
        files = os.listdir(path)
        for file in files:
            if file[-5:] != ".html":
                continue
            with open(os.path.join(path, file), 'r') as f:
                html = f.read()
                soup = bs4.BeautifulSoup(html, 'html.parser')
                table_list += soup.findAll('table')
    return table_list


def convert_html_tables_to_latex(html_path, tables, ids, output_path):
    final_latex_str = ""
    output_new_path = os.path.join(output_path, "figure", "new_result_vis")
    os.makedirs(output_new_path, exist_ok=True)
    for table in tables:
        id = table.findAll('tr')[4].td.p.next_sibling.text
        if id in ids:
            latex_str = LATEX_TEMPLATE
            titles = table.findAll('p')
            dataset_name = "nyu"
            del titles[3]
            del titles[5]
            del titles[5]
            for title in titles:
                latex_str = latex_str.replace('example-text', title.text.strip(), 1)
                if title.text == "MP3D-mesh rendered depth":
                    dataset_name = "mp3d"

            images = table.findAll('img')
            standard_img_size = Image.open(os.path.join(html_path, images[0]['src'])).size
            for img_index, img in enumerate(images):
                img_name = dataset_name + '_' + id.replace(".png", "") + '_' + str(img_index) + ".png"
                img_path = os.path.join(html_path, img['src'])
                dest_img_path = os.path.join(output_new_path, img_name)
                tmp_img = Image.open(img_path)
                if tmp_img.size[0] == tmp_img.size[1]:
                    new_img = resize_and_add_padding(tmp_img, standard_img_size)
                    new_img.save(dest_img_path)
                else:
                    copyfile(img_path, dest_img_path)
                latex_str = latex_str.replace('example-image', os.path.join("figure", "new_result_vis", img_name), 1)
            final_latex_str = final_latex_str + latex_str

    with open(os.path.join(output_path, "main.tex"), 'w') as f:
        f.write(LATEX_HEADER + final_latex_str + LATEX_FOOTER)
    z = zipfile.ZipFile('latex_tables.zip', 'w')
    for path, dirname, filenames in os.walk(output_path):
        file_path = path.replace(output_path, "")
        for filename in filenames:
            z.write(os.path.join(path, filename), os.path.join(file_path, filename))
    z.close()
    rmtree(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--html_folder', help="The folder consists of HTMLs with tables")
    parser.add_argument('--image_ids', help="E.g., 1003.png 1004.png", nargs='+')
    parser.add_argument('--output_path', default="output")
    args = parser.parse_args()
    tables = extract_table_tags_from_html_and_merge(args.html_folder)
    convert_html_tables_to_latex(args.html_folder, tables, args.image_ids, args.output_path)
    print("Done. Latex project saved at " + os.path.abspath("latex_tables.zip"))
