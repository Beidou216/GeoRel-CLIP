import os
from typing import Any, Tuple

import torch
import torchvision.datasets


class GenericDataset(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform, test=False, **kwargs) -> None:
        assert (root is not None) and (transform is not None)
        train = not test
        if train:
            split_data_dir = os.path.join(root, "train")
        else:
            split_data_dir = os.path.join(root, "test")

        # Define a function to check if a file is valid (exclude hidden files/directories)
        def is_valid_file(filepath):
            # Skip hidden files and directories (like .ipynb_checkpoints)
            if any(part.startswith('.') for part in filepath.split(os.sep)):
                return False
            return True

        super(GenericDataset, self).__init__(
            root=split_data_dir,
            transform=transform,
            is_valid_file=is_valid_file
        )


class Aircraft(torch.utils.data.Dataset):
    """FGVC Aircraft dataset with proper train/test split"""

    def __init__(self, transform, test=True, **kwargs):
        self.root = "/root/autodl-tmp/dataset/Aircraft"
        self.data_dir = os.path.join(self.root, "data")
        self.variants_file = os.path.join(self.root, "variants.txt")
        self.transform = transform
        self.test = test

        # Read variants to create class name mapping
        self.variant_to_id = {}
        with open(self.variants_file, 'r') as f:
            for idx, line in enumerate(f):
                variant = line.strip()
                self.variant_to_id[variant] = idx

        # Create all possible class names (100 classes total)
        self.classes = []
        for idx in range(100):  # FGVC Aircraft has 100 classes
            if idx in self.variant_to_id.values():
                variant = [k for k, v in self.variant_to_id.items() if v == idx][0]
                class_name = f"class_{idx:03d}_{variant}"
            else:
                class_name = f"class_{idx:03d}_unknown"
            self.classes.append(class_name)

        # Determine split file
        if test:
            split_file = os.path.join(self.root, "images_variant_test.txt")
        else:
            split_file = os.path.join(self.root, "images_variant_trainval.txt")

        # Load samples
        self.samples = []
        with open(split_file, 'r') as f:
            for line in f:
                # 使用 maxsplit=1 因为 variant 名称可能包含空格（如 'Saab 340'）
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    img_name = parts[0]
                    variant = parts[1]

                    if variant in self.variant_to_id:
                        class_id = self.variant_to_id[variant]
                        img_path = os.path.join(self.data_dir, img_name + ".jpg")
                        if os.path.exists(img_path):
                            self.samples.append((img_path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]

        from torchvision.datasets.folder import default_loader
        image = default_loader(img_path)

        if self.transform:
            image = self.transform(image)

        return image, class_id


class Bird(torch.utils.data.Dataset):
    """CUB-200-2011 dataset with proper train/test split and bbox cropping"""

    def __init__(self, transform, test=True, use_bbox_crop=True, **kwargs):
        self.root = "/root/autodl-tmp/dataset/CUB_200_2011/CUB_200_2011"
        self.images_dir = os.path.join(self.root, "images")
        self.transform = transform
        self.test = test
        self.use_bbox_crop = use_bbox_crop
        
        # 读取类别信息
        classes_file = os.path.join(self.root, "classes.txt")
        self.classes = []
        class_id_to_idx = {}
        with open(classes_file, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    class_id = int(parts[0]) - 1  # 转为 0-based
                    class_name = parts[1]
                    self.classes.append(class_name)
                    class_id_to_idx[class_id] = class_id
        
        # 读取图片路径和类别标签
        images_file = os.path.join(self.root, "images.txt")
        image_paths = {}
        with open(images_file, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    img_id = int(parts[0])
                    img_path = parts[1]
                    image_paths[img_id] = os.path.join(self.images_dir, img_path)
        
        # 读取图片-类别映射
        labels_file = os.path.join(self.root, "image_class_labels.txt")
        image_labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_id = int(parts[0])
                    class_id = int(parts[1]) - 1  # 转为 0-based
                    image_labels[img_id] = class_id
        
        # 读取 bounding boxes
        self.bboxes = {}
        if self.use_bbox_crop:
            bbox_file = os.path.join(self.root, "bounding_boxes.txt")
            if os.path.exists(bbox_file):
                with open(bbox_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            img_id = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            self.bboxes[img_id] = (x, y, width, height)
        
        # 读取训练/测试分割
        split_file = os.path.join(self.root, "train_test_split.txt")
        split_ids = set()
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_id = int(parts[0])
                    is_train = int(parts[1])
                    # is_train: 1=训练集, 0=测试集
                    if (test and is_train == 0) or (not test and is_train == 1):
                        split_ids.add(img_id)
        
        # 构建样本列表
        self.samples = []
        for img_id in sorted(split_ids):
            if img_id in image_paths and img_id in image_labels:
                img_path = image_paths[img_id]
                class_id = image_labels[img_id]
                if os.path.exists(img_path):
                    bbox = self.bboxes.get(img_id, None) if self.use_bbox_crop else None
                    self.samples.append((img_path, class_id, img_id, bbox))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_id, img_id, bbox = self.samples[idx]
        
        from torchvision.datasets.folder import default_loader
        from PIL import Image
        
        image = default_loader(img_path)
        
        # 如果有 bbox，进行裁剪
        if bbox is not None:
            x, y, width, height = bbox
            img_width, img_height = image.size
            
            # 转为整数并确保在图片范围内
            left = max(0, int(x))
            top = max(0, int(y))
            right = min(img_width, int(x + width))
            bottom = min(img_height, int(y + height))
            
            # 确保裁剪区域有效
            if right > left and bottom > top:
                crop = image.crop((left, top, right, bottom))
                
                # 信箱式缩放：按长边=224等比缩放，短边居中填充
                scale = 224 / max(crop.width, crop.height)
                new_w, new_h = int(crop.width * scale), int(crop.height * scale)
                resized = crop.resize((new_w, new_h), Image.BICUBIC)
                
                # 创建224x224画布，居中粘贴
                canvas = Image.new("RGB", (224, 224), (0, 0, 0))
                paste_left = (224 - new_w) // 2
                paste_top = (224 - new_h) // 2
                canvas.paste(resized, (paste_left, paste_top))
                image = canvas
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_id


class Car(torchvision.datasets.ImageFolder):
    """
    Car数据集，需要调整类别名称格式以匹配标准文件
    """

    def __init__(self, transform, test=True, **kwargs):
        # Car数据集有train/test分割，使用test目录
        if test:
            root_path = "/root/autodl-tmp/dataset/archive/car_data/car_data/test"
        else:
            root_path = "/root/autodl-tmp/dataset/archive/car_data/car_data/train"
        
        # Define a function to check if a file is valid (exclude hidden files/directories)
        def is_valid_file(filepath):
            # Skip hidden files and directories (like .ipynb_checkpoints)
            if any(part.startswith('.') for part in filepath.split(os.sep)):
                return False
            return True
        
        # 调用父类初始化
        super(Car, self).__init__(
            root=root_path,
            transform=transform,
            is_valid_file=is_valid_file
        )
        
        # 调整类别名称格式以匹配标准文件
        self._adjust_class_names()
    
    def _adjust_class_names(self):
        """
        调整类别名称格式：从 'Brand Model Year' 改为 'Year Brand Model'
        以匹配标准类别名称文件格式
        """
        # 读取标准类别名称文件
        standard_classes = []
        with open('/root/autodl-tmp/data/classnames/stanford_cars.txt', 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        class_id, class_name = parts
                        standard_classes.append(class_name)
        
        # 创建从原始类别名到标准类别名的映射
        class_mapping = {}
        for original_class in self.classes:
            # 尝试匹配标准类别名称
            for standard_class in standard_classes:
                # 提取年份和主要部分进行匹配
                original_parts = original_class.split()
                standard_parts = standard_class.split()
                
                # 如果年份在最后，提取年份
                if original_parts[-1].isdigit():
                    year = original_parts[-1]
                    brand_model = ' '.join(original_parts[:-1])
                    
                    # 处理特殊字符差异：C-V vs C/V
                    brand_model_normalized = brand_model.replace('C-V', 'C/V')
                    standard_normalized = standard_class.replace('C/V', 'C-V')
                    
                    # 检查是否匹配标准格式
                    if (f"{year} {brand_model}" == standard_class or 
                        f"{year} {brand_model_normalized}" == standard_class or
                        f"{year} {brand_model}" == standard_normalized):
                        class_mapping[original_class] = standard_class
                        break
        
        # 更新类别名称和顺序
        if class_mapping:
            # 按照标准文件的顺序重新排列类别
            new_classes = []
            new_class_to_idx = {}
            new_samples = []
            
            # 创建从标准类别到原始类别的反向映射
            reverse_mapping = {v: k for k, v in class_mapping.items()}
            
            # 按照标准文件的顺序重新排列
            for i, standard_class in enumerate(standard_classes):
                if standard_class in reverse_mapping:
                    original_class = reverse_mapping[standard_class]
                    new_classes.append(standard_class)
                    new_class_to_idx[standard_class] = i
                    
                    # 找到原始类别在数据集中的索引
                    original_idx = self.classes.index(original_class)
                    
                    # 更新样本标签：将原始索引映射到新的索引
                    for sample_path, label in self.samples:
                        if label == original_idx:
                            new_samples.append((sample_path, i))
            
            # 更新属性
            self.classes = new_classes
            self.class_to_idx = new_class_to_idx
            self.samples = new_samples


class Caltech101(torchvision.datasets.ImageFolder):
    """
    Caltech-101数据集，没有train/test分割，直接使用所有数据
    只保留标准类别名称文件中的100个类别
    """

    def __init__(self, transform, test=True, **kwargs):
        # Caltech-101没有train/test分割，所有数据都在101_ObjectCategories/101_ObjectCategories目录下
        root_path = "/root/autodl-tmp/dataset/caltech-101/caltech-101/101_ObjectCategories/101_ObjectCategories"
        
        # Define a function to check if a file is valid (exclude hidden files/directories)
        def is_valid_file(filepath):
            # Skip hidden files and directories (like .ipynb_checkpoints)
            if any(part.startswith('.') for part in filepath.split(os.sep)):
                return False
            return True
        
        super(Caltech101, self).__init__(
            root=root_path,
            transform=transform,
            is_valid_file=is_valid_file
        )
        
        # 只保留标准类别名称文件中的类别
        self._filter_standard_classes()

    def _filter_standard_classes(self):
        """按照 caltech101.txt 的顺序重新排列类别"""
        classnames_file = "/root/autodl-tmp/data/classnames/caltech101.txt"
        if not os.path.exists(classnames_file):
            return
        
        # 读取标准类别顺序
        standard_classes = []
        with open(classnames_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    _, class_name = line.split('\t', 1)
                    # 将下划线替换为空格（与零样本分类一致）
                    standard_classes.append(class_name.replace("_", " "))
        
        # 保存原始的类别到索引的映射
        old_class_to_idx = self.class_to_idx.copy()
        
        # 创建原始类别名（带下划线）到处理后类别名（空格）的映射
        original_to_processed = {cls: cls.replace("_", " ") for cls in self.classes}
        processed_to_original = {v: k for k, v in original_to_processed.items()}
        
        # 按照标准顺序重新构建
        new_classes = []
        new_class_to_idx = {}
        old_to_new_id = {}  # 旧ID -> 新ID的映射
        
        for new_id, std_class in enumerate(standard_classes):
            # 查找对应的原始类别
            if std_class in processed_to_original:
                orig_class = processed_to_original[std_class]
                if orig_class in old_class_to_idx:
                    old_id = old_class_to_idx[orig_class]
                    new_classes.append(std_class)
                    new_class_to_idx[std_class] = new_id
                    old_to_new_id[old_id] = new_id
        
        # 更新类别信息
        self.classes = new_classes
        self.class_to_idx = new_class_to_idx
        
        # 重新映射样本标签
        new_samples = []
        for img_path, old_id in self.samples:
            if old_id in old_to_new_id:
                new_id = old_to_new_id[old_id]
                new_samples.append((img_path, new_id))
        
        self.samples = new_samples


class DTD(torch.utils.data.Dataset):
    """DTD 数据集，默认使用 fold 1 的测试集"""

    def __init__(self, transform, test=True, fold=1, **kwargs):
        self.root = "/root/autodl-tmp/dataset/dtd-r1.0.1/dtd"
        self.images_dir = os.path.join(self.root, "images")
        self.labels_dir = os.path.join(self.root, "labels")
        self.transform = transform
        self.test = test
        self.fold = fold
        
        # 读取类别（从 images 目录的子目录）
        self.classes = sorted([d for d in os.listdir(self.images_dir) 
                              if os.path.isdir(os.path.join(self.images_dir, d)) 
                              and not d.startswith('.')])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 确定使用哪个分割文件
        if test:
            split_file = os.path.join(self.labels_dir, f"test{fold}.txt")
        else:
            split_file = os.path.join(self.labels_dir, f"train{fold}.txt")
        
        # 读取分割文件
        self.samples = []
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                for line in f:
                    img_path = line.strip()
                    if img_path:
                        # 从路径提取类别名（格式：类别名/图片名.jpg）
                        parts = img_path.split('/')
                        if len(parts) >= 2:
                            class_name = parts[0]
                            if class_name in self.class_to_idx:
                                class_id = self.class_to_idx[class_name]
                                full_path = os.path.join(self.images_dir, img_path)
                                if os.path.exists(full_path):
                                    self.samples.append((full_path, class_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        
        from torchvision.datasets.folder import default_loader
        image = default_loader(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_id


class EuroSAT(torchvision.datasets.ImageFolder):
    """
    EuroSAT数据集，没有train/test分割，直接使用所有数据
    """

    def __init__(self, transform, test=True, **kwargs):
        # EuroSAT数据集没有train/test分割，所有数据都在EuroSAT_RGB目录下
        root_path = "/root/autodl-tmp/dataset/EuroSAT_RGB/EuroSAT_RGB"
        
        # Define a function to check if a file is valid (exclude hidden files/directories)
        def is_valid_file(filepath):
            # Skip hidden files and directories (like .ipynb_checkpoints)
            if any(part.startswith('.') for part in filepath.split(os.sep)):
                return False
            return True
        
        super(EuroSAT, self).__init__(
            root=root_path,
            transform=transform,
            is_valid_file=is_valid_file
        )
        
        # 调整类别名称格式以匹配标准文件
        self._adjust_class_names()
    
    def _adjust_class_names(self):
        """
        调整类别名称格式：从驼峰命名改为描述性命名
        以匹配标准类别名称文件格式
        """
        # 读取标准类别名称文件
        standard_classes = []
        with open('/root/autodl-tmp/data/classnames/eurosat.txt', 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        class_id, class_name = parts
                        standard_classes.append(class_name)
        
        # 创建从原始类别名到标准类别名的映射
        class_mapping = {
            'AnnualCrop': 'Annual Crop Land',
            'Forest': 'Forest',
            'HerbaceousVegetation': 'Herbaceous Vegetation Land',
            'Highway': 'Highway or Road',
            'Industrial': 'Industrial Buildings',
            'Pasture': 'Pasture Land',
            'PermanentCrop': 'Permanent Crop Land',
            'Residential': 'Residential Buildings',
            'River': 'River',
            'SeaLake': 'Sea or Lake'
        }
        
        # 更新类别名称和顺序
        if class_mapping:
            # 按照标准文件的顺序重新排列类别
            new_classes = []
            new_class_to_idx = {}
            new_samples = []
            
            # 按照标准文件的顺序重新排列
            for i, standard_class in enumerate(standard_classes):
                # 找到对应的原始类别名
                original_class = None
                for orig, std in class_mapping.items():
                    if std == standard_class:
                        original_class = orig
                        break
                
                if original_class and original_class in self.classes:
                    new_classes.append(standard_class)
                    new_class_to_idx[standard_class] = i
                    
                    # 找到原始类别在数据集中的索引
                    original_idx = self.classes.index(original_class)
                    
                    # 更新样本标签：将原始索引映射到新的索引
                    for sample_path, label in self.samples:
                        if label == original_idx:
                            new_samples.append((sample_path, i))
            
            # 更新属性
            self.classes = new_classes
            self.class_to_idx = new_class_to_idx
            self.samples = new_samples


class Food(torch.utils.data.Dataset):
    """Food-101 数据集，使用官方的 train/test 分割"""

    def __init__(self, transform, test=True, **kwargs):
        self.root = "/root/autodl-tmp/dataset/food-101"
        self.images_dir = os.path.join(self.root, "images")
        self.meta_dir = os.path.join(self.root, "meta")
        self.transform = transform
        self.test = test
        
        # 读取类别（从 images 目录的子目录）
        self.classes = sorted([d for d in os.listdir(self.images_dir) 
                              if os.path.isdir(os.path.join(self.images_dir, d)) 
                              and not d.startswith('.')])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 确定使用哪个分割文件
        if test:
            split_file = os.path.join(self.meta_dir, "test.txt")
        else:
            split_file = os.path.join(self.meta_dir, "train.txt")
        
        # 读取分割文件
        self.samples = []
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                for line in f:
                    img_path = line.strip()
                    if img_path:
                        # 格式：类别名/图片名（如 apple_pie/134679）
                        parts = img_path.split('/')
                        if len(parts) == 2:
                            class_name = parts[0]
                            img_name = parts[1]
                            
                            if class_name in self.class_to_idx:
                                class_id = self.class_to_idx[class_name]
                                full_path = os.path.join(self.images_dir, class_name, img_name + ".jpg")
                                if os.path.exists(full_path):
                                    self.samples.append((full_path, class_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        
        from torchvision.datasets.folder import default_loader
        image = default_loader(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_id


class Flower(torch.utils.data.Dataset):
    """Oxford Flowers dataset with proper train/test split"""

    def __init__(self, transform, test=True, **kwargs):
        self.root = "/root/autodl-tmp/dataset/OxfordFlower102"
        self.images_dir = os.path.join(self.root, "jpg")
        self.setid_file = os.path.join(self.root, "setid.mat")
        self.labels_file = os.path.join(self.root, "imagelabels.mat")
        self.transform = transform
        self.test = test

        # Load set IDs and labels using scipy
        from scipy.io import loadmat
        setid = loadmat(self.setid_file)
        labels = loadmat(self.labels_file)

        # Extract data
        if test:
            # Test set (setid['tstid'][0] contains 1-based indices)
            image_ids = setid['tstid'][0] - 1  # Convert to 0-based
        else:
            # Train set (setid['trnid'][0] contains 1-based indices)
            image_ids = setid['trnid'][0] - 1  # Convert to 0-based

        # Get labels (labels['labels'][0] contains 1-based class indices)
        all_labels = labels['labels'][0] - 1  # Convert to 0-based

        # Create samples
        self.samples = []
        for img_id in image_ids:
            img_path = os.path.join(self.images_dir, f"image_{img_id + 1:05d}.jpg")
            if os.path.exists(img_path):
                label = all_labels[img_id]
                self.samples.append((img_path, label))

        # Create class names (102 classes)
        self.classes = []
        for i in range(102):
            self.classes.append(f"class_{i:03d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        from torchvision.datasets.folder import default_loader
        image = default_loader(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(self, transform, test=False, **kwargs):
        root = "/root/autodl-tmp/dataset/imagenet"
        train = not test
        if train:
            split_data_dir = os.path.join(root, "train", "train")
        else:
            split_data_dir = os.path.join(root, "val", "val")

        # Define a function to check if a file is valid (exclude hidden files/directories)
        def is_valid_file(filepath):
            # Skip hidden files and directories (like .ipynb_checkpoints)
            if any(part.startswith('.') for part in filepath.split(os.sep)):
                return False
            return True

        super(ImageNet, self).__init__(
            root=split_data_dir,
            transform=transform,
            is_valid_file=is_valid_file
        )


class Pet(torchvision.datasets.DatasetFolder):
    """Oxford Pets dataset with proper train/test split"""

    def __init__(self, transform, test=True, **kwargs):
        # Oxford Pets dataset structure: images/ and annotations/
        root = "/root/autodl-tmp/dataset/OxfordPets"
        images_dir = os.path.join(root, "images")
        annotations_dir = os.path.join(root, "annotations")

        # Create a temporary directory structure for ImageFolder
        import tempfile
        import shutil

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

        # Read class names from the classname file
        classname_file = "/root/autodl-tmp/data/classnames/oxford_pets.txt"
        id_to_classname = {}
        with open(classname_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    class_id = int(parts[0])
                    class_name = parts[1]
                    id_to_classname[class_id] = class_name

        # Read the appropriate split file
        if test:
            split_file = os.path.join(annotations_dir, "test.txt")
        else:
            split_file = os.path.join(annotations_dir, "trainval.txt")

        # Read split file and organize by class
        class_images = {}
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_id = int(parts[1]) - 1  # Convert from 1-based to 0-based indexing

                    # Use the correct class name from the mapping
                    if class_id in id_to_classname:
                        class_name = id_to_classname[class_id]

                        if class_name not in class_images:
                            class_images[class_name] = []
                        class_images[class_name].append(img_name)

        # Create class directories and copy files
        for class_name, img_list in class_images.items():
            class_dir = os.path.join(self.temp_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Copy images for this class
            for img_name in img_list:
                src = os.path.join(images_dir, img_name + ".jpg")
                dst = os.path.join(class_dir, img_name + ".jpg")
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        # Now use the temporary directory with ImageFolder
        from torchvision.datasets.folder import default_loader
        super(Pet, self).__init__(root=self.temp_dir, loader=default_loader, extensions=('.jpg', '.jpeg', '.png'),
                                  transform=transform)

    def __del__(self):
        # Clean up temporary directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)


class PetWithText(torchvision.datasets.DatasetFolder):
    """Pet dataset that returns image-text pairs for training"""

    def __init__(self, transform, tokenizer, test=False, **kwargs):
        root = "/root/autodl-tmp/dataset/OxfordPets"
        images_dir = os.path.join(root, "images")
        annotations_dir = os.path.join(root, "annotations")

        # Create a temporary directory structure for ImageFolder
        import tempfile
        import shutil

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

        # Read class names from the classname file
        classname_file = "/root/autodl-tmp/data/classnames/oxford_pets.txt"
        id_to_classname = {}
        with open(classname_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    class_id = int(parts[0])
                    class_name = parts[1]
                    id_to_classname[class_id] = class_name

        # Read the appropriate split file
        if test:
            split_file = os.path.join(annotations_dir, "test.txt")
        else:
            split_file = os.path.join(annotations_dir, "trainval.txt")

        # Read split file and organize by class
        class_images = {}
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_id = int(parts[1]) - 1  # Convert from 1-based to 0-based indexing

                    # Use the correct class name from the mapping
                    if class_id in id_to_classname:
                        class_name = id_to_classname[class_id]

                        if class_name not in class_images:
                            class_images[class_name] = []
                        class_images[class_name].append(img_name)

        # Create class directories and copy files
        for class_name, img_list in class_images.items():
            class_dir = os.path.join(self.temp_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Copy images for this class
            for img_name in img_list:
                src = os.path.join(images_dir, img_name + ".jpg")
                dst = os.path.join(class_dir, img_name + ".jpg")
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        # Now use the temporary directory with ImageFolder
        from torchvision.datasets.folder import default_loader
        super(PetWithText, self).__init__(root=self.temp_dir, loader=default_loader,
                                          extensions=('.jpg', '.jpeg', '.png'), transform=transform)
        self.tokenizer = tokenizer

        # Create class name mapping
        self.class_names = [cls.replace('_', ' ') for cls in self.classes]

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        class_name = self.class_names[label]
        text = f"a photo of a {class_name}"
        text_input = self.tokenizer(text)
        return image, text_input

    def __del__(self):
        # Clean up temporary directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)


class SUN397(torch.utils.data.Dataset):
    """SUN397 数据集，使用 Partitions 的 10-fold 划分，默认 fold 1"""

    def __init__(self, transform, test=True, fold=1, **kwargs):
        # 数据集根目录（包含 Partitions 和 SUN397 子目录）
        dataset_base = "/root/autodl-tmp/dataset/SUN397"
        # 实际图片数据在 SUN397/SUN397/ 子目录下
        self.root = os.path.join(dataset_base, "SUN397")
        # Partitions 目录在数据集根目录下
        self.partitions_dir = os.path.join(dataset_base, "Partitions")
        self.transform = transform
        self.test = test
        self.fold = fold
        
        # 读取类别名称
        classname_file = os.path.join(self.partitions_dir, "ClassName.txt")
        self.classes = []
        if os.path.exists(classname_file):
            with open(classname_file, 'r') as f:
                for line in f:
                    class_name = line.strip()
                    if class_name:
                        # 去掉开头的 / 并转换为标准格式
                        if class_name.startswith('/'):
                            class_name = class_name[1:]
                        self.classes.append(class_name)
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 确定使用哪个分割文件
        if test:
            split_file = os.path.join(self.partitions_dir, f"Testing_{fold:02d}.txt")
        else:
            split_file = os.path.join(self.partitions_dir, f"Training_{fold:02d}.txt")
        
        # 读取分割文件
        self.samples = []
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                for line in f:
                    img_path = line.strip()
                    if img_path:
                        # 路径格式：/场景类别/图片名.jpg
                        if img_path.startswith('/'):
                            img_path = img_path[1:]  # 去掉开头的 /
                        
                        # 从路径提取场景类别
                        parts = img_path.split('/')
                        if len(parts) >= 2:
                            # 场景名可能是 letter/scene 或 letter/scene/subtype
                            if len(parts) == 2:
                                scene = parts[0]
                            else:
                                scene = '/'.join(parts[:-1])
                            
                            if scene in self.class_to_idx:
                                class_id = self.class_to_idx[scene]
                                full_path = os.path.join(self.root, img_path)
                                if os.path.exists(full_path):
                                    self.samples.append((full_path, class_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        
        from torchvision.datasets.folder import default_loader
        image = default_loader(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_id


class SUN397_Old(torchvision.datasets.ImageFolder):
    """旧的 SUN397 实现（已废弃）"""
    
    def _rebuild_dataset_old(self):
        """
        重新构建数据集，根据实际数据集结构匹配标准类别名称
        """
        # 读取标准类别名称文件
        standard_classes = []
        with open('/root/autodl-tmp/data/classnames/sun397.txt', 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        class_id, class_name = parts
                        standard_classes.append(class_name)
        
        # 创建从数据集路径到标准类别名称的映射
        path_to_standard = {}
        
        # 遍历所有字母目录
        for letter_dir in self.classes:  # 字母目录如 'a', 'b', 'c' 等
            letter_path = os.path.join("/root/autodl-tmp/dataset/SUN397", letter_dir)
            if os.path.exists(letter_path):
                for scene_name in os.listdir(letter_path):
                    scene_path = os.path.join(letter_path, scene_name)
                    if os.path.isdir(scene_path) and not scene_name.startswith('.'):
                        # 检查是否有子类别
                        subdirs = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d)) and not d.startswith('.')]
                        
                        if subdirs:
                            # 有子类别，处理每个子类别
                            for subdir in subdirs:
                                # 构建数据集路径格式
                                dataset_path = f"{letter_dir}/{scene_name}/{subdir}"
                                
                                # 尝试多种匹配方式
                                possible_names = [
                                    f"{subdir} {scene_name}",  # outdoor apartment_building
                                    f"{scene_name} {subdir}",  # apartment_building outdoor
                                    f"{subdir}_{scene_name}",  # outdoor_apartment_building
                                    f"{scene_name}_{subdir}",  # apartment_building_outdoor
                                ]
                                
                                # 在标准类别中查找匹配
                                for possible_name in possible_names:
                                    if possible_name in standard_classes:
                                        path_to_standard[dataset_path] = possible_name
                                        break
                        else:
                            # 没有子类别，直接匹配场景名
                            dataset_path = f"{letter_dir}/{scene_name}"
                            if scene_name in standard_classes:
                                path_to_standard[dataset_path] = scene_name
        
        # 按照标准文件的顺序重新构建数据集
        new_classes = []
        new_class_to_idx = {}
        new_samples = []
        
        for i, standard_class in enumerate(standard_classes):
            # 查找匹配的数据集路径
            matching_paths = [path for path, std_class in path_to_standard.items() if std_class == standard_class]
            
            if matching_paths:
                new_classes.append(standard_class)
                new_class_to_idx[standard_class] = i
                
                # 收集该标准类别的所有样本
                for dataset_path in matching_paths:
                    # 从原始样本中找到匹配的样本
                    for sample_path, label in self.samples:
                        if dataset_path.replace('/', os.sep) in sample_path:
                            new_samples.append((sample_path, i))
        
        # 更新属性
        self.classes = new_classes
        self.class_to_idx = new_class_to_idx
        self.samples = new_samples


class UCF101(GenericDataset):

    def __init__(self, transform, test=True, **kwargs):
        super(UCF101, self).__init__(
            root="/dataset/UCF101",
            transform=transform,
            test=test,
        )
