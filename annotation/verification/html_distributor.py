import bs4
import os
import argparse

def video_embed(soup, new_sub_div, view_link):
    front_video = soup.new_tag("video")
    front_video["class"] = "lazy-video"
    front_video["controls"] = "True"
    front_video["autoplay"] = "True"
    front_video["muted"] = "True"
    front_video["loop"] = "True"
    front_video["src"] = ""
    new_sub_div.append(front_video)

    new_link = soup.new_tag("source")
    new_link["data-src"] = view_link
    new_link["type"] = "video/mp4"
    front_video.append(new_link)
    
def insert_html_pattern(soup, video_id, front_mesh_link, front_point_link, top_point_link):
    new_div = soup.new_tag("div")
    new_div['id'] = "videoal"
    soup.body.append(new_div)

    new_sub_text_div = soup.new_tag("div")
    new_sub_text_div["class"] = "text"
    new_div.append(new_sub_text_div)

    new_text = soup.new_tag("b")
    new_text.string = video_id
    new_sub_text_div.append(new_text)

    new_sub_front_div = soup.new_tag("div")
    new_sub_front_div["class"] = "video"
    new_div.append(new_sub_front_div)

    new_sub_top_div = soup.new_tag("div")
    new_sub_top_div["class"] = "video"
    new_div.append(new_sub_top_div)

    new_sub_front_mesh_div = soup.new_tag("div")
    new_sub_front_mesh_div["class"] = "video"
    new_div.append(new_sub_front_mesh_div)

    video_embed(soup, new_sub_front_div, front_point_link)
    video_embed(soup, new_sub_top_div, top_point_link)
    video_embed(soup, new_sub_front_mesh_div, front_mesh_link)

def swap_domain(link):
    sub_urls = link.split("/")
    temp = sub_urls[-2]
    sub_urls[-2] = sub_urls[-3]
    sub_urls[-3] = temp
    link = "/".join(sub_urls)
    return link

def export_html(template_path, save_path, video_folder, video_file_list):
    # load the file
    with open(template_path) as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt, features="html.parser")
    if args.mesh_topdown == True:
        front_mesh_folder = os.path.join(video_folder, "video_meshplane_topdown")
    else:
        front_mesh_folder = os.path.join(video_folder, "video_meshplane_front")
    front_point_folder = os.path.join(video_folder, "video_pointplane_front")
    top_point_folder = os.path.join(video_folder, "video_pointplane_topdown")

    for video_file in video_file_list:    
        video_id = video_file.split(".")[0]
        front_mesh_link = os.path.join(front_mesh_folder, video_file)
        front_point_link = os.path.join(front_point_folder, video_file)
        top_point_link = os.path.join(top_point_folder, video_file)
        swap_domain(front_mesh_link)
        if args.disp == "err":
            front_mesh_link = swap_domain(front_mesh_link)
            front_point_link = swap_domain(front_point_link)
            top_point_link = swap_domain(top_point_link)
            
        insert_html_pattern(soup, video_id, front_mesh_link, front_point_link, top_point_link)
    # save the file again
    with open(save_path, "w") as outf:
        outf.write(str(soup))

if __name__ == "__main__":
    # link_list = read_txt()
    parser = argparse.ArgumentParser(description='Get Setting')
    parser.add_argument('--dataset_folder_path', default="", type=str)
    parser.add_argument('--disp', default="", type=str)
    parser.add_argument('--mesh_topdown', default=False, action='store_true', help='Bool type')
    parser.add_argument('--page_len', default=20, type=int)

    args = parser.parse_args()

    video_folder = "../../verification/video/{}".format(args.dataset_folder_path)
    if args.mesh_topdown == True:
        front_mesh_folder = os.path.join(video_folder, "video_meshplane_topdown")
    else:
        front_mesh_folder = os.path.join(video_folder, "video_meshplane_front")


    html_folder = "../../verification/html/{}".format(args.dataset_folder_path)
    os.makedirs(html_folder, exist_ok=True)
    
    if args.disp == "err":
        video_path_list = []
        for subfolder in os.listdir(video_folder):
            video_sub_path_list = os.listdir(os.path.join(video_folder, subfolder, "video_meshplane_topdown"))
            video_path_list += [os.path.join(subfolder, path) for path in video_sub_path_list]
    else:
        video_path_list = os.listdir(front_mesh_folder)
    video_chunks = [video_path_list[x:x+args.page_len] for x in range(0, len(video_path_list), args.page_len)]
    template_path = "../../verification/html/template/template.html"
    for idx, video_file_list in enumerate(video_chunks):
        video_folder_for_html = video_folder.replace("/verification", "")
        save_path = os.path.join(html_folder, "{}.html".format(idx))
        export_html(template_path, save_path, video_folder_for_html, video_file_list)


    
    

