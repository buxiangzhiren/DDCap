import torch
import skimage.io as io
from clip1 import clip
from clip1.clip import _transform
from PIL import Image
import pickle
import json
import os
import time
from tqdm import tqdm
import argparse
from PIL import ImageFile
# import clip
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from misc import format_seconds


class CocoDataset(Dataset):
    def __init__(self, preprocess):
        with open('./data/coco/annotations/train_caption.json', 'r') as f:
            self.data = json.load(f)
        self.preprocess = preprocess
        print("length of the dataset is ")
        print(len(self.data))

        self.num = len(self.data)


    def __len__(self):
        return self.num

    def __getitem__(self, index):
        d = self.data[index]
        d["clip_embedding"] = index
        img_id = d["image_id"]
        filename = f"/zzx_vlexp/VQ-Diffusion-my2/MSCOCO_Caption/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"/zzx_vlexp/VQ-Diffusion-my2/MSCOCO_Caption/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = io.imread(filename)
        image = self.preprocess(Image.fromarray(image))
        return d, image


def main(clip_model_type: str):
    device = torch.device('cuda:1')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"/zzx_vlexp/CLIP_prefix_caption/data/coco/oscar_split_{clip_model_name}_train.pkl"
    clip_model, _ = clip.load(clip_model_type, device=device, jit=False)
    # clip_model.to(device)
    with open('./data/coco/annotations/train_caption.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    preprocess= _transform(224)
    dataset = CocoDataset(preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=False,  # (val_sampler is None),
                                             num_workers=1,
                                             pin_memory=True,
                                             sampler=None,
                                             # drop_last=True,
                                             persistent_workers=True)
    step_start = time.time()
    for itr, (d, image) in enumerate(dataloader):
        itr_start = time.time()
        batch_size = image.size()[0]
        image = image.to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        for i in range(batch_size):
            dt = {}
            all_embeddings.append(prefix[i,:,:].unsqueeze(0))
            dt['image_id'] = d['image_id'][i]
            dt['id'] = d['id'][i]
            dt['caption'] = d['caption'][i]
            dt['clip_embedding'] = d['clip_embedding'][i]
            all_captions.append(dt)
        val_iters = len(dataset) // batch_size
        info = 'iter {}/{}'.format(itr, val_iters)
        itr_time_avg = (time.time() - step_start) / (itr + 1)
        info += ' || iter_time: {it}s | left_time: {lt}'.format(
            it=round(time.time() - itr_start, 1),
            lt=format_seconds(itr_time_avg * (val_iters - itr - 1))
        )
        print(info)
        if (itr + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)






    # for i in tqdm(range(len(data))):
    #     d = data[i]
    #     img_id = d["image_id"]
    #     filename = f"/data3/zzx/MSCOCO_Caption/train2014/train2014/COCO_train2014_{int(img_id):012d}.jpg"
    #     if not os.path.isfile(filename):
    #         filename = f"/data3/zzx/MSCOCO_Caption/val2014/val2014/COCO_val2014_{int(img_id):012d}.jpg"
    #     image = io.imread(filename)
    #     image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         prefix = clip_model.encode_image(image).cpu()
    #     d["clip_embedding"] = i
    #     all_embeddings.append(prefix)
    #     all_captions.append(d)
    #     if (i + 1) % 10 == 0:
    #         with open(out_path, 'wb') as f:
    #             pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
