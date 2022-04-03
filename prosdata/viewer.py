import torch
import ipywidgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from itertools import chain, islice
from ipywidgets import interactive, widgets

def _create_label(text:str)->ipywidgets.widgets.Label:
    "Create label widget"
    
    label = widgets.Label(
            text,
            layout=widgets.Layout(
            width='100%', 
            display='flex', 
            justify_content="center"
            )
        )
    return label
        
def _create_slider(
    slider_min: int, 
    slider_max: int, 
    value: int, 
    step: int=1, 
    description:str ='',
    continuous_update: bool=True, 
    readout: bool=False, 
    slider_type: str='IntSlider',
    **kwargs)->ipywidgets.widgets:
    "Create slider widget"
    
    slider = getattr(widgets, slider_type)(
        min=slider_min, 
        max=slider_max, 
        step=step, 
        value=value,
        description=description, 
        continuous_update=continuous_update,
        readout = readout,
        layout=widgets.Layout(width='99%', min_width='200px'),
        style={'description_width': 'initial'}, 
        **kwargs
    )   
    return slider
    
def _create_button(description:str)->ipywidgets.widgets.Button:
    "Create button widget"
    button = widgets.Button(
            description=description,
            layout=widgets.Layout(
                width='95%',
                margin='5px 5px'
            )
        )
    return button

def _create_togglebutton(description: str, 
                         value: int, 
                         **kwargs)->ipywidgets.widgets.Button:
    "Create toggle button widget"
    button = widgets.ToggleButton(
            description=description,
            value = value,
            layout=widgets.Layout(
                width='95%',
                margin='5px 5px'
            ), **kwargs
        )
    return button


class BasicViewer():
    """ Base class for viewing TensorDicom3D objects. 
    
    Args: 
        x: main image object to view as rank 3 tensor
        y: either a segmentation mask as as rank 3 tensor or a label as str. 
        prediction: a class predicton as str
        description: description of the whole image
        figsize: size of image, passed as plotting argument
        cmap: colormap for the image
    Returns: 
        Instance of BasicViewer
    """
    
    def __init__(self, x:torch.Tensor, y=None, prediction:str=None, description: str=None, 
                 figsize=(3, 3), cmap:str='bone'):
        assert x.ndim == 3, f"x.ndim needs to be equal to but is {x.ndim}"
        if isinstance(y, torch.Tensor): 
            assert x.shape == y.shape, f"Shapes of x {x.shape} and y {y.shape} do not match"
        self.x=x
        self.y=y
        self.prediction=prediction
        self.description=description
        self.figsize=figsize
        self.cmap=cmap
        self.with_mask = isinstance(y, torch.Tensor)
        self.slice_range = (1, len(x)) # len(x) == im.shape[0]
                            
    def _plot_slice(self, im_slice, with_mask, px_range):
        "Plot slice of image"
        fig, ax = plt.subplots(1, 1, figsize=self.figsize) 
        ax.imshow(self.x[im_slice-1, :, :].clip(*px_range), cmap=self.cmap)
        if isinstance(self.y, (torch.Tensor)) and with_mask:
            ax.imshow(self.y[im_slice-1, :, :], cmap='jet', alpha = 0.25)
        plt.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
    
    def _create_image_box(self, figsize):
        "Create widget items, order them in item_box and generate view box"
        items = []
        
        if self.description: plot_description = _create_label(self.description)
        
        if isinstance(self.y, str):
            label = f'{self.y} | {self.prediction}' if self.prediction else self.y
            if self.prediction: 
                font_color = 'green' if self.y == self.prediction else 'red'
                y_label = _create_label(r'\(\color{' + font_color + '} {' + label  + '}\)') 
            else: 
                y_label = _create_label(label)
        else: y_label = _create_label(' ')
            
        slice_slider = _create_slider(
            slider_min = min(self.slice_range), 
            slider_max = max(self.slice_range), 
            value = max(self.slice_range)//2, 
            readout = True)

        toggle_mask_button = _create_togglebutton('Show Mask', True)
        
        range_slider = _create_slider(
            slider_min = self.x.min().numpy(), 
            slider_max = self.x.max().numpy(),  
            value = [self.x.min().numpy(), self.x.max().numpy()], 
            slider_type = 'FloatRangeSlider' if torch.is_floating_point(self.x) else 'IntRandSlider', 
            step = 0.01 if torch.is_floating_point(self.x) else 1, 
            readout=True)
        
        image_output = widgets.interactive_output(
            f = self._plot_slice,
            controls = {'im_slice': slice_slider, 
                'with_mask': toggle_mask_button, 
                'px_range': range_slider})

        image_output.layout.height = f'{self.figsize[0]/1.2}in' # suppress flickering
        image_output.layout.width = f'{self.figsize[1]/1.2}in' # suppress flickering
        
        if self.description: items.append(plot_description)
        items.append(y_label)
        items.append(range_slider)        
        items.append(image_output)
        if isinstance(self.y, torch.Tensor): 
            slice_slider = widgets.HBox([slice_slider, toggle_mask_button])     
        items.append(slice_slider)        
                    
        image_box=widgets.VBox(
            items,
            layout = widgets.Layout(
                border = 'none', 
                margin = '10px 5px 0px 0px', 
                padding =  '5px'))
        
        return image_box
    
    def _generate_views(self):
        image_box = self._create_image_box(self.figsize)
        self.box = widgets.HBox(children=[image_box])

    @property
    def image_box(self):
        return self._create_image_box(self.figsize)
    
    def show(self):
        self._generate_views()
        plt.style.use('default')
        display(self.box)
        
        
class DicomExplorer(BasicViewer):
    """ DICOM viewer for basic image analysis inside iPython notebooks. 
    Can display a single 3D volume together with a segmentation mask, a histogram 
    of voxel/pixel values and some summary statistics. 
    Allows simple windowing by clipping the pixel/voxel values to a region, which 
    can be manually specified.    
    
    """
    
    vbox_layout = widgets.Layout(
        margin = '10px 5px 5px 5px', 
        padding =  '5px',
        display='flex',
        flex_flow='column',
        align_items='center', 
        min_width = '250px')
            
    def _plot_hist(self, px_range):
        x = self.x.numpy().flatten()
        fig, ax = plt.subplots(figsize=self.figsize)
        N, bins, patches = plt.hist(x, 100, color='grey')
        lwr = int(px_range[0] * 100/max(x))
        upr = int(np.ceil(px_range[1] * 100/max(x)))
        
        for i in range(0,lwr):
            patches[i].set_facecolor('grey' if lwr > 0 else 'darkblue')
        for i in range(lwr, upr):
            patches[i].set_facecolor('darkblue')
        for i in range(upr,100):
            patches[i].set_facecolor('grey' if upr < 100 else 'darkblue')
        
        plt.show()
        
    def _image_summary(self, px_range):
        x = self.x.clip(*px_range)

        diffs = x - x.mean()
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0))
        kurt = torch.mean(torch.pow(zscores, 4.0)) - 3.0

        table = f'Statistics:\n' + \
                f'  Mean px value:     {x.mean()} \n' + \
                f'  Std of px values:  {x.std()} \n' + \
                f'  Min px value:      {x.min()} \n' + \
                f'  Max px value:      {x.max()} \n' + \
                f'  Median px value:   {x.median()} \n' + \
                f'  Skewness:          {skews} \n' + \
                f'  Kurtosis:          {kurt} \n\n' + \
                f'Tensor properties \n' + \
                f'  Tensor shape:      {tuple(x.shape)}\n' + \
                f'  Tensor dtype:      {x.dtype}'
        print(table)
        
    def _generate_views(self):   
        
        slice_slider = _create_slider(
            slider_min = min(self.slice_range), 
            slider_max = max(self.slice_range), 
            value = max(self.slice_range)//2, 
            readout = True)

        toggle_mask_button = _create_togglebutton('Show Mask', True)
        
        range_slider = _create_slider(
            slider_min = self.x.min().numpy(), 
            slider_max = self.x.max().numpy(),  
            value = [self.x.min().numpy(), self.x.max().numpy()], 
            continuous_update=False,
            slider_type = 'FloatRangeSlider' if torch.is_floating_point(self.x) else 'IntRandSlider', 
            step = 0.01 if torch.is_floating_point(self.x) else 1)
        
        image_output = widgets.interactive_output(
            f = self._plot_slice,
            controls = {'im_slice': slice_slider, 
                'with_mask': toggle_mask_button, 
                'px_range': range_slider})
                        
        image_output.layout.height = f'{self.figsize[0]/1.2}in' # suppress flickering
        image_output.layout.width = f'{self.figsize[1]/1.2}in' # suppress flickering
        
        if isinstance(self.y, torch.Tensor): 
            slice_slider = widgets.HBox([slice_slider, toggle_mask_button])     
        
        hist_output = widgets.interactive_output(
            f = self._plot_hist,
            controls = {'px_range': range_slider})
        
        hist_output.layout.height = f'{self.figsize[0]/1.2}in' # suppress flickering
        hist_output.layout.width = f'{self.figsize[1]/1.2}in' # suppress flickering

        toggle_mask_button = _create_togglebutton('Show Mask', True)
        
        table_output = widgets.interactive_output(
            f = self._image_summary, 
            controls = {'px_range': range_slider})
        
        table_box = widgets.VBox([table_output], layout=self.vbox_layout)
        
        hist_box = widgets.VBox(
            [hist_output, range_slider],
            layout=self.vbox_layout)
        
        image_box = widgets.VBox(
            [image_output, slice_slider],
            layout=self.vbox_layout)
        
        self.box = widgets.HBox(
            [image_box, hist_box, table_box], 
            layout = widgets.Layout(
                border = 'solid 1px lightgrey', 
                margin = '10px 5px 0px 0px', 
                padding =  '5px', 
                width = f'{self.figsize[1]*2 + 3}in'))
        
        
class ListViewer(object):
    """ Display multipple images with their masks or labels/predictions.
    Arguments: 
        x (tuple, list): Tensor objects to view
        y (tuple, list): Tensor objects (in case of segmentation task) or class labels as string. 
        predictions (str): Class predictions 
        cmap: colormap for display of `x`
        max_n: maximum number of items to display
    """
    
    def __init__(self, x:(list, tuple), y=None, prediction:str=None, description: str=None, 
                 figsize=(4, 4), cmap:str='bone', max_n = 9):
        self.slice_range = (1, len(x))      
        x = x[0:max_n]
        if y: y = y[0:max_n]
        self.x=x
        self.y=y
        self.prediction=prediction
        self.description=description
        self.figsize=figsize
        self.cmap=cmap
        self.max_n=max_n
        
    def _generate_views(self):
        n_images = len(self.x)
        image_grid, image_list = [], []
        
        for i in range(0, n_images):
            image = self.x[i]
            mask = self.y[i] if isinstance(self.y, list) else None
            pred = self.prediction[i] if self.prediction else None
            
            image_list.append(
                BasicViewer(
                    x = image, 
                    y = mask, 
                    prediction = pred, 
                    figsize = self.figsize, 
                    cmap = self.cmap)
                .image_box)
            
            if (i+1) % np.ceil(np.sqrt(n_images)) == 0 or i == n_images - 1: 
                image_grid.append(widgets.HBox(image_list))
                image_list = []
        
        self.box = widgets.VBox(children=image_grid) 
        
    def show(self):
        self._generate_views()
        plt.style.use('default')
        display(self.box)