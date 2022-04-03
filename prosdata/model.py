# create a standard UNet

from monai.networks.nets import UNet

def get_model(config: dict):
    return UNet(
        spatial_dims=config.ndim,
        in_channels=len(config.data.image_cols),
        out_channels=config.model.out_channels,
        channels=config.model.channels,
        strides=config.model.strides,
        num_res_units=config.model.num_res_units,
        act=config.model.act,
        norm=config.model.norm,
        dropout=config.model.dropout,
            )
