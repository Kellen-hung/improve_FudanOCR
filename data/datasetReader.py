import lmdb
import six
import sys
import random
import torchvision.transforms as transforms
import os
import cv2
import numpy as np

from PIL import Image
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset

def preprocess_inference_image(img_path):
    """推論時的固定前處理，保持與訓練一致"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # # === (1) 去除背景不均 (保留淡筆畫) ===
    # bg = cv2.GaussianBlur(img, (51, 51), 0)
    # img = cv2.addWeighted(img, 1.5, bg, -0.5, 0)
    # cv2.imwrite('./test01.png', img)

    # # # === (2) 局部對比增強 CLAHE ===
    # # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # # img = clahe.apply(img)
    # # cv2.imwrite('./test02.png', img)

    # # === (3) 輕微銳化 ===
    # blur_small = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.addWeighted(img, 1.2, blur_small, -0.2, 0)
    # cv2.imwrite('./test03.png', img)

    # # === (4) 限制範圍 ===
    # img = np.clip(img, 0, 255).astype(np.uint8)
    # cv2.imwrite('./test04.png', img)

    # # === (5) 轉 PIL Image 給 transform 用 ===
    return Image.fromarray(img).convert("RGB")

class InferenceDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        # if you don't need sort, just delete it
        self.fnames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))])
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.img_dir, fname)
        img = preprocess_inference_image(path)
        if self.transform:
            img = self.transform(img)
        return img, fname

class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, reverse=False, alphabet=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.reverse = reverse

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index > len(self):
            index = len(self) - 1
        assert index <= len(self), 'index range error index: %d' % index
        # index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())
            
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')

                pass
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))

            label = strQ2B(label)
            label += '$'
            label = label.lower()
            if self.transform is not None:
                img = self.transform(img)
        return (img, label, index)

def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

class resizeNormalize(object):

    def __init__(self, size, test=False, interpolation=Image.BILINEAR):
        self.test = test
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        width, height = img.size
        if 1.5 * width < height:
            img = img.transpose(Image.ROTATE_90)
        img = img.resize(self.size, self.interpolation)

        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class IndependentHalvesSampler(Sampler):
    def __init__(self, dataset, batch_size, iteration_num):

        len_dict = {}
        self.dataset = dataset
        self.batch_size = batch_size
        self.iteration_num = iteration_num

        for i in range(len(dataset)):
            img, label = dataset[i]
            if len(label) not in len_dict.keys():
                len_dict[len(label)] = [i]
            else:
                len_dict[len(label)].append(i)

        self.len_dict = len_dict

    def __iter__(self):
        batch = []
        assert (self.batch_size % 16 == 0)

        for i in range(self.iteration_num):
            for i in range(2, 10):
                list_index = self.len_dict[i]
                for j in range(2):
                    index = random.randint(0, len(list_index) - 1)
                    batch.append(list_index[index])
        return iter(batch)

    def __len__(self):
        return self.iteration_num * self.batch_size
