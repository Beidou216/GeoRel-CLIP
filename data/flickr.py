import os
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class TextListWrapper:
    """
    A wrapper that makes a list of text lists behave like both a tensor and a list.
    This is needed because zeroshot_retrieval.py expects batch_texts to be:
    - A tensor (line 64: batch_texts.to(device))
    - A list of lists (line 66: for ind, texts in zip(inds, batch_texts))
    """
    def __init__(self, texts_list):
        self.texts_list = texts_list
        # Flatten and stack all tokenized texts into a tensor
        all_texts = []
        for texts in texts_list:
            if isinstance(texts, (list, tuple)):
                all_texts.extend(texts)
            else:
                all_texts.append(texts)
        self.tensor = torch.stack(all_texts) if len(all_texts) > 0 and isinstance(all_texts[0], torch.Tensor) else None
    
    def to(self, device):
        """Make it compatible with batch_texts.to(device) - update internal tensor and return self"""
        if self.tensor is not None:
            self.tensor = self.tensor.to(device)
        return self  # Return the wrapper itself, not the tensor
    
    def __iter__(self):
        """Make it iterable for line 66: for ind, texts in zip(inds, batch_texts)"""
        return iter(self.texts_list)
    
    def __getitem__(self, idx):
        """Allow indexing"""
        return self.texts_list[idx]
    
    def __len__(self):
        return len(self.texts_list)


def flickr_collate_fn(batch):
    """
    Custom collate function for Flickr datasets.
    Returns a TextListWrapper that behaves like both a tensor and a list of lists.
    """
    images, texts_list = zip(*batch)
    images = torch.stack(images)
    batch_texts = TextListWrapper(texts_list)
    return images, batch_texts


class Flickr8K(Dataset):
    """Flickr8K dataset for image-text retrieval evaluation"""

    def __init__(self, tokenizer=None, transform=None, test=True, split=None, **kwargs):
        """
        Args:
            tokenizer: Optional tokenizer function. If None, returns raw text strings.
            transform: Optional image transform. If None, returns PIL Image.
            test: If True, use test split; if False, use train split.
            split: Explicit split name ("train", "dev", "test"). Overrides test parameter.
        """
        super().__init__()
        
        self.root = "/root/autodl-tmp/dataset/Flickr8K"
        self.images_dir = os.path.join(self.root, "Images")
        self.text_data_dir = os.path.join(self.root, "Flickr_TextData")
        
        self.tokenizer = tokenizer
        self.transform = transform
        self.test = test
        
        # Determine split
        if split is None:
            split = "test" if test else "train"
        self.split = split
        
        # Load split file
        split_file = os.path.join(self.text_data_dir, f"Flickr_8k.{split}Images.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        # Read image filenames for this split
        self.image_filenames = []
        with open(split_file, 'r') as f:
            for line in f:
                filename = line.strip()
                if filename:
                    self.image_filenames.append(filename)
        
        # Load all captions from token file
        token_file = os.path.join(self.text_data_dir, "Flickr8k.token.txt")
        if not os.path.exists(token_file):
            raise FileNotFoundError(f"Token file not found: {token_file}")
        
        # Build image to captions mapping
        self.image_to_captions = {}
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Format: image_filename#caption_id\tcaption_text
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    continue
                
                image_caption_id = parts[0].strip()
                caption_text = parts[1].strip()
                
                # Extract image filename (remove #caption_id)
                if '#' in image_caption_id:
                    image_filename = image_caption_id.split('#')[0]
                else:
                    continue
                
                if image_filename not in self.image_to_captions:
                    self.image_to_captions[image_filename] = []
                self.image_to_captions[image_filename].append(caption_text)
        
        # Filter to only include images in the split that have captions
        self.samples = []
        for filename in self.image_filenames:
            if filename in self.image_to_captions:
                img_path = os.path.join(self.images_dir, filename)
                if os.path.exists(img_path):
                    self.samples.append((img_path, self.image_to_captions[filename]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, captions = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Process captions
        if self.tokenizer is None:
            # Return list of raw text strings
            texts = captions
        else:
            # Tokenize each caption
            texts = [self.tokenizer(caption) for caption in captions]
        
        return image, texts


class Flickr30K(Dataset):
    """Flickr30K dataset for image-text retrieval evaluation"""

    def __init__(self, tokenizer=None, transform=None, test=True, split=None, **kwargs):
        """
        Args:
            tokenizer: Optional tokenizer function. If None, returns raw text strings.
            transform: Optional image transform. If None, returns PIL Image.
            test: If True, use test split; if False, use train split.
            split: Explicit split name ("train", "val", "test"). Overrides test parameter.
        """
        super().__init__()
        
        self.root = "/root/autodl-tmp/dataset/Flickr30K"
        self.images_dir = os.path.join(self.root, "flickr30k_images")
        self.text_data_dir = self.root  # captions.txt is in root directory
        
        self.tokenizer = tokenizer
        self.transform = transform
        self.test = test
        
        # Determine split
        if split is None:
            split = "test" if test else "train"
        self.split = split
        
        # Load split file - try common Flickr30K split file names
        split_candidates = [
            os.path.join(self.root, f"Flickr_30k.{split}Images.txt"),
            os.path.join(self.root, f"{split}Images.txt"),
            os.path.join(self.root, f"flickr30k_{split}.txt"),
            os.path.join(self.root, f"{split}.txt")
        ]
        
        split_file = None
        for candidate in split_candidates:
            if os.path.exists(candidate):
                split_file = candidate
                break
        
        if split_file is None:
            raise FileNotFoundError(f"Split file not found for {split}: {', '.join(split_candidates)}")
        
        # Read image filenames for this split
        self.image_filenames = []
        with open(split_file, 'r') as f:
            for line in f:
                filename = line.strip()
                if filename:
                    # Remove path if present, keep only filename
                    filename = os.path.basename(filename)
                    self.image_filenames.append(filename)
        
        # Load all captions from captions.txt
        captions_file = os.path.join(self.root, "captions.txt")
        if not os.path.exists(captions_file):
            raise FileNotFoundError(f"Captions file not found: {captions_file}")
        
        # Build image to captions mapping
        self.image_to_captions = {}
        import csv
        with open(captions_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                if len(row) >= 3:
                    image_filename = row[0].strip()
                    caption_text = row[2].strip()
                    
                    if image_filename not in self.image_to_captions:
                        self.image_to_captions[image_filename] = []
                    self.image_to_captions[image_filename].append(caption_text)
        
        # Filter to only include images in the split that have captions
        self.samples = []
        for filename in self.image_filenames:
            if filename in self.image_to_captions:
                img_path = os.path.join(self.images_dir, filename)
                if os.path.exists(img_path):
                    self.samples.append((img_path, self.image_to_captions[filename]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, captions = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Process captions
        if self.tokenizer is None:
            # Return list of raw text strings
            texts = captions
        else:
            # Tokenize each caption
            texts = [self.tokenizer(caption) for caption in captions]
        
        return image, texts

