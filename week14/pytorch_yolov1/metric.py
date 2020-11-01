from data.evaluate.voc_eval import voc_evaluation
import torch
import dist_comm as comm
from apex.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


class base_val(object):
    def __init__(self, img_info, gt_info, classes):
        self.img_info = img_info
        self.gt_info = gt_info
        self.classes = classes

    def get_img_info(self, x):
        return self.img_info[x]

    def get_groundtruth(self, x):
        return self.gt_info[x]

    def map_class_id_to_class_name(self, x):
        return self.classes[x]


def training_eval(model, valload, classes, device):
    model.eval()
    img_info = dict()
    gt_info = dict()
    predictions = []
    print("eval {} total img {}".format(len(classes), len(valload.dataset)))
    with torch.no_grad():
        for img, meta in tqdm(valload):

            bs = img.shape[0]
            img = img.to(device)
            batch_box = model(img)
            assert len(batch_box) == bs, (len(batch_box))

            for i in range(bs):
                fileID = meta[i]['fileID']
                img_info[fileID] = dict(width=meta[i]['img_width'], height=meta[i]['img_height'])
                gt_info[fileID] = meta[i]['boxlist']
                box = batch_box[i]

                # TODO padding test
                box.resize((meta[i]['img_width'], meta[i]['img_height']))
                predictions.append([fileID, box])

    all_info = comm.all_gather(img_info)
    all_gt = comm.all_gather(gt_info)
    all_pred = comm.all_gather(predictions)
    comm.synchronize()
    if comm.is_main_process():
        for i in range(1, len(all_info)):
            all_info[0].update(all_info[i])
            all_gt[0].update(all_gt[i])
            all_pred[0] += all_pred[i]
        print(len(all_info[0].keys()))
        gt_sets = base_val(all_info[0], all_gt[0], classes)
        result = voc_evaluation(gt_sets, all_pred[0], './', box_only=True)
        for i, value in enumerate(result['ap']):
            result[classes[i]] = value
        del result['ap']
    else:
        result = []

    model.train()

    return result


import argparse


def parse():
    parser = argparse.ArgumentParser(description='PyTorch Detector Training')
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    return args


args = parse()

if __name__ == "__main__":
    from data.build import make_dist_voc_loader
    from cfg.voc import cfg
    from yolo import create_yolov1
    import os

    train_cfg = cfg['train_cfg']
    model_cfg = cfg['model_cfg']
    model_name = model_cfg['model_type']
    epochs = train_cfg['epochs']
    classes = train_cfg['classes']
    bs = train_cfg['batch_size']
    device = train_cfg['device']
    out_dir = train_cfg['out_dir']
    train_root = train_cfg['dataroot']
    patch_size = train_cfg['patch_size']

    out_dir = out_dir + '/' + model_name

    model = create_yolov1(model_cfg)

    checkpoint = torch.load('{}/best_model.pth'.format(out_dir))['model']
    data_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}


    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)

    model.load_state_dict(data_dict, strict=True)
    model.eval()
    model.cuda()
    model = DDP(model)
    valloader = make_dist_voc_loader(os.path.join(train_root, 'VOC2007_test.txt'), img_size=[(448, 448)],
                                     batch_size=16,
                                     train=False,
                                     )
    training_eval(model, valloader, classes, device)
