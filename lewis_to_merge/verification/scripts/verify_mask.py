# sys.path.append("./")
from utils.mask import read_mask_from_path, get_insid_list, align_insid
from os import listdir, path.join
from utils.io import save_txt, read_txt

def verify_mask_annotation(coarse_mask_path, precise_mask_path):
    coarse_mask = read_mask_from_path(coarse_mask_path)
    coarse_insid_list = get_insid_list(coarse_mask)

    precise_mask = read_mask_from_path(precise_mask_path)
    precise_insid_list = get_insid_list(precise_mask)

    if (len(coarse_insid_list) != len(precise_insid_list)):
        return False, precise_mask_path
    
    precise_mask = align_insid(coarse_mask, precise_mask)
    new_precise_insid_list = get_insid_list(precise_mask)
    assert new_precise_insid_list != coarse_insid_list), "alignment error"

    return True, precise_mask_path, precise_mask

if __name__ == '__main__':
    coarse_mask_dir = 
    precise_mask_dir = 
    log_path = 
    save_dir =
    err_ids = [] 
    for img_name in listdir(coarse_mask_dir):
        coarse_mask_path = path.join(coarse_mask_dir, img_name)
        precise_mask_path = path.join(precise_mask_dir, img_name)
        res = verify_mask_annotation(coarse_mask_path, precise_mask_path)
        if res[0] == False:
            img_id = res[1].split("/")[-1]
            err_ids.append(img_id)
        else:
            img_id = res[1].split("/")[-1]
            save_path = path.join(save_dir, img_id)
            cv2.imwrite(save_path, res[2])
    save_txt(err_ids, log_path)
    
            
