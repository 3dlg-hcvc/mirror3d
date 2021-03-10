import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from mirror3d_resnet import resnet50
from classifier_Dataset import *
from utils.general_utlis import *
from tqdm import tqdm
import operator

def eval_get_score(args, val_loader, criterion, model):
    img_score = dict()
    for i, (img_path_list, images, _) in enumerate(tqdm(val_loader)):
        
        output = model(images)
        output = torch.nn.functional.softmax(output,dim=1)

        obj_score = output[:,-1]
        for index, img_path in enumerate(img_path_list):
            img_score[img_path] = obj_score[index].item()

    img_score = dict(sorted(img_score.items(), key=operator.itemgetter(1),reverse=True))
    json_save_path = os.path.join(args.json_output_save_folder, "imgPath_score_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())  + ".json")
    save_json(json_save_path,img_score)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--unsort_img_list',  default="", type=str)
    parser.add_argument('--resume_path',  default="", type=str) 
    parser.add_argument('--json_output_save_folder', default="", type=str)
    args = parser.parse_args()

    print(args)

    
    os.makedirs(args.json_output_save_folder, exist_ok=True)
    


    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    model = resnet50()
    model.eval()
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                momentum=0.9,
                                weight_decay=1e-4)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
    # used for visulize image
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )


    train_sampler = None
    val_dataset = Dataset_to_label(args.unsort_img_list,transform) 
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=1, pin_memory=True, sampler=train_sampler)


    model = torch.nn.DataParallel(model).cuda()


    if args.resume_path:
        print("=> loading checkpoint '{}'".format(args.resume_path))
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume_path, checkpoint['epoch']))
    else:
        print("------- wihout pretrained checkpoint")
    cudnn.benchmark = True

    eval_get_score(args, val_loader, criterion, model)