import bs4
import os
import argparse
import zipfile
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
\\begin{figure}
\\setkeys{Gin}{width=\\linewidth}
\\begin{tabularx}{\\textwidth}{Y Y@{\\hspace{1mm}} Y@{\\hspace{0mm}} Y@{\\hspace{1mm}} Y@{\\hspace{1mm}} Y@{\\hspace{2mm}} | Y@{\\hspace{1mm}} Y@{\\hspace{1mm}} Y@{\\hspace{2mm}} | Y@{\\hspace{1mm}} Y@{\\hspace{1mm}} Y@{\\hspace{0mm}} }
 & & & \\multicolumn{3}{c}{\\textcolor{black!60!green}{\\scriptsize example-text}} & \\multicolumn{3}{c}{\\textcolor{blue!60!cyan}{\\scriptsize example-text}} & \\multicolumn{3}{c}{\\textcolor{cyan}{\\scriptsize example-text}}\\\\
& \\noindent\\parbox[t]{\\hsize}{\\fontsize{6}{6}\\selectfont Input Depth (D)} &  & \\vspace{4pt}\\tikzmarknode{A1}{\\includegraphics{example-image}} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\tikzmarknode{A2}{\\includegraphics{example-image}} & \\vspace{4pt}\\tikzmarknode{B1}{\\includegraphics{example-image}} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\tikzmarknode{B2}{\\includegraphics{example-image}} & \\vspace{4pt}\\tikzmarknode{C1}{\\includegraphics{example-image}} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\tikzmarknode{C2}{\\includegraphics{example-image}}
\\\\ &
\\vspace{4pt}\\includegraphics{example-image} & \\multicolumn{1}{r}{\\tiny example-text} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} \\\\\\cline{2-12} & \\vspace{4pt}\\includegraphics{example-image} & \\multicolumn{1}{r}{\\tiny example-text} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} \\\\\\cline{4-12}
& \\noindent\\parbox[b]{\\hsize}{\\fontsize{6}{6}\\selectfont Color (RGB)} & \\multicolumn{1}{r}{\\tiny example-text} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} & \\vspace{4pt}\\includegraphics{example-image} \\\\
 & & & \\tiny Depth & \\tiny Error & \\tiny PC & \\tiny Depth & \\tiny Error & \\tiny PC & \\tiny Depth & \\tiny Error & \\tiny PC \\\\
 & & & \\multicolumn{3}{c}{\\scriptsize example-text} & \\multicolumn{3}{c}{\\scriptsize example-text} & \\multicolumn{3}{c}{\\scriptsize example-text}\\\\

\\end{tabularx}
\\begin{tikzpicture}[overlay,remember picture]
\\node[ draw=black!60!green, line width=1pt, fit={(A1)(A2)($(A1.north west)+(+2pt,-0pt)$)($(A2.south east)+(-2pt,-0pt)$)}]{};
\\node[ draw=blue!60!cyan, line width=1pt,fit={(B1)(B2)($(B1.north west)+(+2pt,-0pt)$)($(B2.south east)+(-2pt,-0pt)$)}]{};
\\node[ draw=cyan, line width=1pt,fit={(C1)(C2)($(C1.north west)+(+2pt,-0pt)$)($(C2.south east)+(-2pt,-0pt)$)}]{};
\\end{tikzpicture}
\\caption{Example figure with tabular graphics.}
\\label{fig:example}
\\end{figure}
'''

LATEX_FOOTER = '''
\\end{document}
'''


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
    counter = 0
    for table in tables:
        id = table.findAll('tr')[4].td.p.next_sibling.text
        if id in ids:
            folder_name = "figure_" + str(counter)
            counter += 1
            output_new_path = os.path.join(output_path, "figure", "result_vis_imgs", folder_name)
            os.makedirs(output_new_path, exist_ok=True)
            latex_str = LATEX_TEMPLATE
            imgs = table.findAll('img')
            for img_index, img in enumerate(imgs):
                img_path = os.path.join(html_path, img['src'])
                dest_img_path = os.path.join(output_new_path, str(img_index) + ".png")
                copyfile(img_path, dest_img_path)
                latex_str = latex_str.replace('example-image', os.path.join("figure", "result_vis_imgs", folder_name, str(img_index) + ".png"), 1)

            titles = table.findAll('p')
            del titles[3]
            del titles[5]
            del titles[5]
            for title in titles:
                latex_str = latex_str.replace('example-text', title.text.strip(), 1)
            final_latex_str = final_latex_str + latex_str

    with open(os.path.join(output_path, "main.tex"), 'w') as f:
        f.write(LATEX_HEADER + final_latex_str + LATEX_FOOTER)
    z = zipfile.ZipFile('latex_tables.zip', 'w')
    for path, dirnames, filenames in os.walk(output_path):
        fpath = path.replace(output_path, "")
        for filename in filenames:
            z.write(os.path.join(path, filename), os.path.join(fpath, filename))
    z.close()
    rmtree("output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--html_folder', help="The folder consists of HTMLs with tables")
    parser.add_argument('--image_ids', help="E.g., 1003.png 1004.png", nargs='+')
    args = parser.parse_args()
    tables = extract_table_tags_from_html_and_merge(args.html_folder)
    convert_html_tables_to_latex(args.html_folder, tables, args.image_ids, "output")
    print("Done. Latex project saved at " + os.path.abspath("latex_tables.zip"))

