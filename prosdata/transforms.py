# create transforms for training, validation and test dataset

## TODO: Make Transforms more dynamic by directly building from config args
## Maybe like this
## TFM_NAME=config.transforms.keys()[0]
## tfm_fun=getattr(monai.transforms, TFM_NAME)
## tmfs+=[tfms_fun(keys=image+cols, **config.transforms[TFM_NAME], prob=prob, mode=mode)


## ---------- imports ----------
import os
# only import of base transforms, others are imported as needed
from monai.utils.enums import CommonKeys
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    ConcatItemsd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    SaveImaged,
    ScaleIntensityd,
    NormalizeIntensityd
)
# images should be interploated with `bilinear` but masks with `nearest`

## ---------- base transforms ----------
# applied everytime
def get_base_transforms(
    config: dict,
    minv: int=0, 
    maxv: int=1
)->list:
    
    tfms=[]
    tfms+=[LoadImaged(keys=config.data.image_cols+config.data.label_cols)]
    tfms+=[EnsureChannelFirstd(keys=config.data.image_cols+config.data.label_cols)]
    if config.transforms.spacing:
        from monai.transforms import Spacingd
        tfms+=[
            Spacingd(
                keys=config.data.image_cols+config.data.label_cols,
                pixdim=config.transforms.spacing,
                mode=config.transforms.mode
            )
        ]
    if config.transforms.orientation:
        from monai.transforms import Orientationd
        tfms+=[
            Orientationd(
                keys=config.data.image_cols+config.data.label_cols,
                axcodes=config.transforms.orientation
            )
        ]
    tfms+=[
        ScaleIntensityd(
            keys=config.data.image_cols,
            minv=minv,
            maxv=maxv
        )
    ]
    tfms+=[NormalizeIntensityd(keys=config.data.image_cols)]
    return tfms

## ---------- train transforms ----------

def get_train_transforms(config: dict):
    tfms=get_base_transforms(config=config)

    # ---------- specific transforms for mri ----------
    if 'rand_bias_field' in config.transforms.keys():
        from monai.transforms import RandBiasFieldd
        args=config.transforms.rand_bias_field
        tfms+=[
            RandBiasFieldd(
                keys=config.data.image_cols,
                degree=args['degree'],
                coeff_range=args['coeff_range'],
                prob=config.transforms.prob
            )
        ]

    if 'rand_gaussian_smooth' in config.transforms.keys():
        from monai.transforms import RandGaussianSmoothd
        args=config.transforms.rand_gaussian_smooth
        tfms+=[
            RandGaussianSmoothd(
                keys=config.data.image_cols,
                sigma_x=args['sigma_x'],
                sigma_y=args['sigma_y'],
                sigma_z=args['sigma_z'],
                prob=config.transforms.prob
            )
        ]

    if 'rand_gibbs_nose' in config.transforms.keys():
        from monai.transforms import RandGibbsNoised
        args=config.transforms.rand_gibbs_nose
        tfms+=[
            RandGibbsNoised(
                keys=config.data.image_cols,
                alpha=args['alpha'],
                prob=config.transforms.prob
            )
        ]

    # ---------- affine transforms ----------

    if 'rand_affine' in config.transforms.keys():
        from monai.transforms import RandAffined
        args=config.transforms.rand_affine
        tfms+=[
            RandAffined(
                keys=config.data.image_cols+config.data.label_cols,
                rotate_range=args['rotate_range'],
                shear_range=args['shear_range'],
                translate_range=args['translate_range'],
                mode=config.transforms.mode,
                prob=config.transforms.prob
            )
        ]

    if 'rand_rotate90' in config.transforms.keys():
        from monai.transforms import RandRotate90d
        args=config.transforms.rand_rotate90
        tfms+=[
            RandRotate90d(
                keys=config.data.image_cols+config.data.label_cols,
                spatial_axes=args['spatial_axes'],
                prob=config.transforms.prob
            )
        ]

    if 'rand_rotate' in config.transforms.keys():
        from monai.transforms import RandRotated
        args=config.transforms.rand_rotate
        tfms+=[
            RandRotated(
                keys=config.data.image_cols+config.data.label_cols,
                range_x=args['range_x'],
                range_y=args['range_y'],
                range_z=args['range_z'],
                mode=config.transforms.mode,
                prob=config.transforms.prob
            )
        ]

    if 'rand_elastic' in config.transforms.keys():
        if config['ndim'] == 3:
            from monai.transforms import Rand3DElasticd as RandElasticd
        elif config['ndim'] == 2:
            from monai.transforms import Rand2DElasticd as RandElasticd
        args=config.transforms.rand_elastic
        tfms+=[
            RandElasticd(
                keys=config.data.image_cols+config.data.label_cols,
                sigma_range=args['sigma_range'],
                magnitude_range=args['magnitude_range'],
                rotate_range=args['rotate_range'],
                shear_range=args['shear_range'],
                translate_range=args['translate_range'],
                mode=config.transforms.mode,
                prob=config.transforms.prob
            )
        ]

    if 'rand_zoom' in config.transforms.keys():
        from monai.transforms import RandZoomd
        args=config.transforms.rand_zoom
        tfms+=[
            RandZoomd(
                keys=config.data.image_cols+config.data.label_cols,
                min_zoom=args['min'],
                max_zoom=args['max'],
                mode=['area' if x == 'bilinear' else x for x in config.transforms.mode],
                prob=config.transforms.prob
            )
        ]

    # ---------- random cropping, very effective for large images ----------
    # RandCropByPosNegLabeld is not advisable for data with missing lables
    # e.g., segmentation of carcinomas which are not present on all images
    # thus fallback to RandSpatialCropSamplesd. Completly replacing Cropping
    # by just resizing could be discussed, but I believe it is not beneficial
    # For the first version, this is an ungly hack. For the second version, 
    # a better verion for transforms should be written. 

    if 'rand_crop_pos_neg_label' in config.transforms.keys():
        from monai.transforms import RandCropByPosNegLabeld
        args=config.transforms.rand_crop_pos_neg_label
        tfms+=[
            RandCropByPosNegLabeld(
                keys=config.data.image_cols+config.data.label_cols,
                label_key=config.data.label_cols[0],
                spatial_size=args['spatial_size'],
                pos=args['pos'],
                neg=args['neg'],
                num_samples=args['num_samples'],
                image_key=config.data.image_cols[0],
                image_threshold=0,
            )
        ]
        
    elif 'rand_spatial_crop_samples' in config.transforms.keys():
        from monai.transforms import RandSpatialCropSamplesd
        args=config.transforms.rand_spatial_crop_samples
        tfms+=[
            RandSpatialCropSamplesd(
                keys=config.data.image_cols+config.data.label_cols,
                roi_size=args['roi_size'],
                random_size=False,
                num_samples=args['num_samples'],
            )
        ]
        
    else: 
        raise ValueError('Either `rand_crop_pos_neg_label` or `rand_spatial_crop_samples` '\
                         'need to be specified')
        
    # ---------- intensity transforms ----------

    if 'gaussian_noise' in config.transforms.keys():
        from monai.transforms import RandGaussianNoised
        args=config.transforms.gaussian_noise
        tfms+=[
            RandGaussianNoised(
                keys=config.data.image_cols,
                mean=args['mean'],
                std=args['std'],
                prob=config.transforms.prob
            )
        ]

    if 'shift_intensity' in config.transforms.keys():
        from monai.transforms import RandShiftIntensityd
        args=config.transforms.shift_intensity
        tfms+=[
            RandShiftIntensityd(
                keys=config.data.image_cols,
                offsets=args['offsets'],
                prob=config.transforms.prob
            )
        ]

    if 'gaussian_sharpen' in config.transforms.keys():
        from monai.transforms import RandGaussianSharpend
        args=config.transforms.gaussian_sharpen
        tfms+=[
            RandGaussianSharpend(
                keys=config.data.image_cols,
                sigma1_x=args['sigma1_x'],
                sigma1_y=args['sigma1_y'],
                sigma1_z=args['sigma1_z'],
                sigma2_x=args['sigma2_x'],
                sigma2_y=args['sigma2_y'],
                sigma2_z=args['sigma2_z'],
                alpha=args['alpha'],
                prob=config.transforms.prob
            )
        ]

    if 'adjust_contrast' in config.transforms.keys():
        from monai.transforms import RandAdjustContrastd
        args=config.transforms.adjust_contrast
        tfms+=[
            RandAdjustContrastd(
                keys=config.data.image_cols,
                gamma=args['gamma'],
                prob=config.transforms.prob
            )
        ]
        
    # Concat mutlisequence data to single Tensors on the ChannelDim
    # Rename images to `CommonKeys.IMAGE` and labels to `CommonKeys.LABELS`
    # for more compatibility with monai.engines
    
    tfms+=[
        ConcatItemsd(
            keys=config.data.image_cols, 
            name=CommonKeys.IMAGE, 
            dim=0
        )
    ]

    tfms+=[
        ConcatItemsd(
            keys=config.data.label_cols, 
            name=CommonKeys.LABEL, 
            dim=0
        )
    ]

    return Compose(tfms)

## ---------- valid transforms ----------

def get_val_transforms(config: dict):
    tfms=get_base_transforms(config=config)
    tfms+=[EnsureTyped(keys=config.data.image_cols+config.data.label_cols)]
    tfms+=[
        ConcatItemsd(
            keys=config.data.image_cols, 
            name=CommonKeys.IMAGE, 
            dim=0
        )
    ]

    tfms+=[
        ConcatItemsd(
            keys=config.data.label_cols, 
            name=CommonKeys.LABEL, 
            dim=0
        )
    ]
    
    return Compose(tfms)

## ---------- test transforms ----------
# same as valid transforms

def get_test_transforms(config: dict):
    tfms=get_base_transforms(config=config)
    tfms+=[EnsureTyped(keys=config.data.image_cols+config.data.label_cols)]
    tfms+=[
        ConcatItemsd(
            keys=config.data.image_cols, 
            name=CommonKeys.IMAGE, 
            dim=0
        )
    ]

    tfms+=[
        ConcatItemsd(
            keys=config.data.label_cols, 
            name=CommonKeys.LABEL, 
            dim=0
        )
    ]
    
    return Compose(tfms)


def get_val_post_transforms(config: dict): 
    tfms=[EnsureTyped(keys=[CommonKeys.PRED, CommonKeys.LABEL]),
            AsDiscreted(
                keys=CommonKeys.PRED, 
                argmax=True, 
                to_onehot=config.model.out_channels, 
                num_classes=config.model.out_channels
            ),
            AsDiscreted(
                keys=CommonKeys.LABEL, 
                to_onehot=config.model.out_channels, 
                num_classes=config.model.out_channels
            ),
            KeepLargestConnectedComponentd(
                keys=CommonKeys.PRED, 
                applied_labels=list(range(1, config.model.out_channels))
           ),
            ]
    return Compose(tfms)
