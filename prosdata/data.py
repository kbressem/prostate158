# create dataloaders form csv file

## ---------- imports ----------
import os        
import torch
import shutil
import numpy as np
import pandas as pd
from typing import Union
from monai.utils import first
from functools import partial
from collections import namedtuple
from monai.data import DataLoader as MonaiDataLoader
      
from . import transforms
from .utils import num_workers


def import_dataset(config: dict): 
    if config.data.dataset_type == 'persistent':
        from monai.data import PersistentDataset
        if os.path.exists(config.data.cache_dir): 
            shutil.rmtree(config.data.cache_dir) # rm previous cache DS
        os.makedirs(config.data.cache_dir, exist_ok = True)
        Dataset = partial(PersistentDataset, cache_dir = config.data.cache_dir)
    elif config.data.dataset_type == 'cache':
        from monai.data import CacheDataset
        raise NotImplementedError('CacheDataset not yet implemented')
    else:
        from monai.data import Dataset
    return Dataset


class DataLoader(MonaiDataLoader): 
    "overwrite monai DataLoader for enhanced viewing capabilities"
    
    def show_batch(self, 
                   image_key: str='image', 
                   label_key: str='label', 
                   image_transform=lambda x: x.squeeze().transpose(0,2).flip(-2), 
                   label_transform=lambda x: x.squeeze().transpose(0,2).flip(-2)): 
        """Args:
            image_key: dict key name for image to view
            label_key: dict kex name for corresponding label. Can be a tensor or str
            image_transform: transform input before it is passed to the viewer to ensure
                ndim of the image is equal to 3 and image is oriented correctly
            label_transform: transform labels before passed to the viewer, to ensure 
                segmentations masks have same shape and orientations as images. Should be 
                identity function of labels are str. 
        """
        from .viewer import ListViewer
        
        batch = first(self)
        image = torch.unbind(batch[image_key], 0)
        label = torch.unbind(batch[label_key], 0)
        
        ListViewer([image_transform(im) for im in image],
                   [label_transform(im) for im in label]).show()

# TODO
## Work with 3 dataloaders
        
def segmentation_dataloaders(config: dict, 
                             train: bool = None,
                             valid: bool = None,
                             test: bool = None,
                            ):
    """Create segmentation dataloaders
    Args:
        config: config file
        train: whether to return a train DataLoader
        valid: whether to return a valid DataLoader
        test: whether to return a test DateLoader
    Args from config: 
        data_dir: base directory for the data
        csv_name: path to csv file containing filenames and paths
        image_cols: columns in csv containing path to images
        label_cols: columns in csv containing path to label files
        dataset_type: PersistentDataset, CacheDataset and Dataset are supported
        cache_dir: cache directory to be used by PersistentDataset
        batch_size: batch size for training. Valid and test are always 1
        debug: run with reduced number of images
    Returns:
        list of:
            train_loader: DataLoader (optional, if train==True)
            valid_loader: DataLoader (optional, if valid==True)
            test_loader: DataLoader (optional, if test==True)
    """
    
    ## parse needed rguments from config
    if train is None: train = config.data.train
    if valid is None: valid = config.data.valid
    if test is None: test = config.data.test
    
    data_dir = config.data.data_dir
    train_csv = config.data.train_csv
    valid_csv = config.data.valid_csv
    test_csv = config.data.test_csv
    image_cols = config.data.image_cols
    label_cols = config.data.label_cols
    dataset_type = config.data.dataset_type
    cache_dir = config.data.cache_dir
    batch_size = config.data.batch_size
    debug = config.debug
            
    ## ---------- data dicts ----------

    # first a global data dict, containing only the filepath from image_cols and label_cols is created. For this,
    # the dataframe is reduced to only the relevant columns. Then the rows are iterated, converting each row into an
    # individual dict, as expected by monai

    if not isinstance(image_cols, (tuple, list)): image_cols = [image_cols]
    if not isinstance(label_cols, (tuple, list)): label_cols = [label_cols]

    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    test_df = pd.read_csv(test_csv)
    if debug: 
        train_df = train_df.sample(25)
        valid_df = valid_df.sample(5)
    
    train_df['split']='train'
    valid_df['split']='valid'
    test_df['split']='test'
    whole_df = []
    if train: whole_df += [train_df]
    if valid: whole_df += [valid_df]
    if test: whole_df += [test_df]
    df = pd.concat(whole_df)
    cols = image_cols + label_cols
    for col in cols:
        # create absolute file name from relative fn in df and data_dir
        df[col] = [os.path.join(data_dir, fn) for fn in df[col]]
        if not os.path.exists(list(df[col])[0]):
            raise FileNotFoundError(list(df[col])[0])


    data_dict = [dict(row[1]) for row in df[cols].iterrows()]
    # data_dict is not the correct name, list_of_data_dicts would be more accurate, but also longer.
    # The data_dict looks like this:
    # [
    #  {'image_col_1': 'data_dir/path/to/image1',
    #   'image_col_2': 'data_dir/path/to/image2'
    #   'label_col_1': 'data_dir/path/to/label1},
    #  {'image_col_1': 'data_dir/path/to/image1',
    #   'image_col_2': 'data_dir/path/to/image2'
    #   'label_col_1': 'data_dir/path/to/label1},
    #    ...]
    # Filename should now be absolute or relative to working directory

    # now we create separate data dicts for train, valid and test data respectively
    assert train or test or valid, 'No dataset type is specified (train/valid or test)'

    if test:
        test_files = list(map(data_dict.__getitem__, *np.where(df.split == 'test')))

    if valid:
        val_files = list(map(data_dict.__getitem__, *np.where(df.split == 'valid')))

    if train:
        train_files = list(map(data_dict.__getitem__, *np.where(df.split == 'train')))

    # transforms are specified in transforms.py and are just loaded here
    if train: train_transforms = transforms.get_train_transforms(config)
    if valid: val_transforms = transforms.get_val_transforms(config)
    if test: test_transforms = transforms.get_test_transforms(config)
    
    
    ## ---------- construct dataloaders ----------
    Dataset=import_dataset(config)
    data_loaders = []
    if train:
        train_ds = Dataset(
            data=train_files,
            transform=train_transforms
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers(),
            shuffle=True
        )
        data_loaders.append(train_loader)

    if valid:
        val_ds = Dataset(
            data=val_files,
            transform=val_transforms
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            num_workers=num_workers(),
            shuffle=False
        )
        data_loaders.append(val_loader)

    if test:
        test_ds = Dataset(
            data=test_files,
            transform=test_transforms
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            num_workers=num_workers(),
            shuffle=False
        )
        data_loaders.append(test_loader)

    # if only one dataloader is constructed, return only this dataloader else return a named tuple with dataloaders,
    # so it is clear which DataLoader is train/valid or test

    if len(data_loaders) == 1:
        return data_loaders[0]
    else:
        DataLoaders = namedtuple(
            'DataLoaders',
            # create str with specification of loader type if train and test are true but
            # valid is false string will be 'train test'
            ' '.join(
                [
                    'train' if train else '',
                    'valid' if valid else '',
                    'test' if test else ''
                ]
            ).strip()
        )
        return  DataLoaders(*data_loaders)
