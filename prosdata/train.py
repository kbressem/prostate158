import os
import yaml
import munch
import torch
import ignite
import monai
import shutil
import pandas as pd

from typing import Union, List, Callable
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from monai.handlers import (
    CheckpointSaver,
    StatsHandler,
    TensorBoardStatsHandler,
    TensorBoardImageHandler,
    ValidationHandler,
    from_engine,
    MeanDice,
    EarlyStopHandler,
    MetricLogger,
    MetricsSaver
)

from .data import segmentation_dataloaders
from .model import get_model
from .optimizer import get_optimizer
from .loss import get_loss
from .transforms import get_val_post_transforms
from .utils import USE_AMP

def loss_logger(engine): 
    "write loss and lr of each iteration/epoch to file"
    iteration=engine.state.iteration
    epoch=engine.state.epoch
    loss=[o['loss'] for o in engine.state.output]
    loss=sum(loss)/len(loss)
    lr=engine.optimizer.param_groups[0]['lr']
    log_file=os.path.join(engine.config.log_dir, 'train_logs.csv')
    if not os.path.exists(log_file): 
        with open(log_file, 'w+') as f: 
            f.write('iteration,epoch,loss,lr\n')
    with open(log_file, 'a') as f: 
        f.write(f'{iteration},{epoch},{loss},{lr}\n')
        
def metric_logger(engine): 
    "write `metrics` after each epoch to file"
    if engine.state.epoch > 1: # only key metric is calcualted in 1st epoch, needs fix
        metric_names=[k for k in engine.state.metrics.keys()]
        metrics=[str(engine.state.metrics[mn]) for mn in metric_names]
        log_file=os.path.join(engine.config.log_dir, 'metric_logs.csv')
        if not os.path.exists(log_file): 
            with open(log_file, 'w+') as f: 
                f.write(','.join(metric_names) + '\n')
        with open(log_file, 'a') as f: 
            f.write(','.join(metrics) + '\n')
        
def pred_logger(engine):
    "save `pred` each time metric improves"
    epoch=engine.state.epoch
    root = os.path.join(engine.config.out_dir, 'preds')
    if not os.path.exists(root): 
        os.makedirs(root) 
        torch.save(
            engine.state.output[0]['label'], 
            os.path.join(root, f'label.pt')
        )
        torch.save(
            engine.state.output[0]['image'], 
            os.path.join(root, f'image.pt')
        )
        
    if epoch==engine.state.best_metric_epoch:
        torch.save(
            engine.state.output[0]['pred'], 
            os.path.join(root, f'pred_epoch_{epoch}.pt')
        )

    
def get_val_handlers(
    network: torch.nn.Module,
    config: dict
    ) -> list:
    """Create default handlers for model validation
    Args: 
        network: 
            nn.Module subclass, the model to train
    
    Returns:
        a list of default handlers for validation: [
            StatsHandler: 
                ???
            TensorBoardStatsHandler: 
                Save loss from validation to `config.log_dir`, allow logging with TensorBoard
            CheckpointSaver: 
                Save best model to `config.model_dir`
        ]
    """
    
    val_handlers=[      
        StatsHandler(
            tag_name="metric_logger",
            epoch_print_logger=metric_logger,
            output_transform=lambda x: None
        ),
        StatsHandler(
            tag_name="pred_logger",
            epoch_print_logger=pred_logger,
            output_transform=lambda x: None
        ),
        TensorBoardStatsHandler(
            log_dir=config.log_dir, 
            # tag_name="val_mean_dice",
            output_transform=lambda x: None
        ),
        TensorBoardImageHandler(
            log_dir=config.log_dir,
            batch_transform=from_engine(["image", "label"]),
            output_transform=from_engine(["pred"]),
        ),
        CheckpointSaver(
            save_dir=config.model_dir, 
            save_dict={f"network_{config.run_id}": network}, 
            save_key_metric=True
        ),

    ]
    
    return val_handlers


def get_train_handlers(
    evaluator: monai.engines.SupervisedEvaluator,
    config: dict
    ) -> list: 
    """Create default handlers for model training
    Args: 
        evaluator: an engine of type `monai.engines.SupervisedEvaluator` for evaluations
        every epoch
        
    Returns:
        list of default handlers for training: [
            ValidationHandler:
                Allows model validation every epoch
            StatsHandler:
                ???
            TensorBoardStatsHandler: 
                Save loss from validation to `config.log_dir`, allow logging with TensorBoard
        ]
    """
    
    train_handlers=[
        ValidationHandler(
            validator=evaluator, 
            interval=1, 
            epoch_level=True
        ),
        StatsHandler(
            tag_name="train_loss", 
            output_transform=from_engine(
                ["loss"], 
                first=True
            )
        ),
        StatsHandler(
            tag_name='loss_logger', 
            iteration_print_logger=loss_logger
        ), 
        TensorBoardStatsHandler(
            log_dir=config.log_dir,
            tag_name="train_loss",
            output_transform=from_engine(
                ["loss"], 
                first=True
            ),
        )
    ]
    
    return train_handlers

def get_evaluator(
    config: dict,
    device: torch.device , 
    network: torch.nn.Module, 
    val_data_loader: monai.data.dataloader.DataLoader, 
    val_post_transforms: monai.transforms.compose.Compose,
    val_handlers: Union[Callable, List]=get_val_handlers
) -> monai.engines.SupervisedEvaluator: 
    
    """Create default evaluator for training of a segmentation model
    Args: 
        device: 
            torch.cuda.device for model and engine
        network: 
            nn.Module subclass, the model to train
        val_data_loader: 
            Validation data loader, `monai.data.dataloader.DataLoader` subclass
        val_post_transforms: 
            function to create transforms OR composed transforms
        val_handlers: 
            function to create handerls OR List of handlers
            
    Returns: 
        default evaluator for segmentation of type `monai.engines.SupervisedEvaluator`
    """
    
    if callable(val_handlers): val_handlers=val_handlers()
    
    evaluator=monai.engines.SupervisedEvaluator(
        device=device,
        val_data_loader=val_data_loader,
        network=network,
        inferer=monai.inferers.SlidingWindowInferer(
            roi_size=(96, 96, 96), 
            sw_batch_size=4, 
            overlap=0.5
        ),
        postprocessing=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=False, 
                output_transform=from_engine(
                    ["pred", "label"]
                )
            )
        },
        val_handlers=val_handlers,
        # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP evaluation
        amp=USE_AMP,
    )
    evaluator.config=config
    return evaluator


class SegmentationTrainer(monai.engines.SupervisedTrainer): 
    "Default Trainer für supervised segmentation task"
    def __init__(self, 
                 config: dict,
                 progress_bar: bool=True, 
                 early_stopping: bool=True, 
                 metrics: list=["MeanDice", "HausdorffDistance", "SurfaceDistance"],
                 save_latest_metrics: bool=True
                ):
        self.config=config
        self._prepare_dirs()
        self.config.device=torch.device(self.config.device)
        
        train_loader, val_loader=segmentation_dataloaders(
            config=config, 
            train=True, 
            valid=True, 
            test=False
        )
        network=get_model(config=config).to(config.device)
        optimizer=get_optimizer(
            network, 
            config=config
        )
        loss_fn=get_loss(config=config)
        val_post_transforms=get_val_post_transforms(config=config)
        val_handlers=get_val_handlers(
            network, 
            config=config
        )
        
        self.evaluator=get_evaluator(
            config=config,
            device=config.device, 
            network=network, 
            val_data_loader=val_loader, 
            val_post_transforms=val_post_transforms,
            val_handlers=val_handlers,

        )
        train_handlers=get_train_handlers(
            self.evaluator, 
            config=config
        )
        
        super().__init__(
            device=config.device, 
            max_epochs=self.config.training.max_epochs, 
            train_data_loader=train_loader,
            network=network,
            optimizer=optimizer, 
            loss_function=loss_fn, 
            inferer=monai.inferers.SimpleInferer(), 
            train_handlers=train_handlers,
            amp=USE_AMP,
        )        
        
        if early_stopping: self._add_early_stopping()
        if progress_bar: self._add_progress_bars()
        
        self.schedulers=[]
        # add different metrics dynamically
        for m in metrics: 
            getattr(monai.handlers, m)(
                include_background=False, 
                reduction="mean", 
                output_transform=from_engine(
                    ["pred", "label"]
                )
            ).attach(self.evaluator, m)
            
        self._add_metrics_logger()
        # add eval loss to metrics
        self._add_eval_loss()
        
        if save_latest_metrics: self._add_metrics_saver()

    
    def _prepare_dirs(self)->None:
        # create run_id, copy config file for reproducibility
        os.makedirs(self.config.run_id, exist_ok=True)
        with open(
            os.path.join(
                self.config.run_id, 
                'config.yaml'
            ), 'w+') as f: 
            f.write(yaml.safe_dump(self.config))
        
        # delete old log_dir
        if os.path.exists(self.config.log_dir): 
            shutil.rmtree(self.config.log_dir) 
            
    def _add_early_stopping(self) -> None: 
        early_stopping=EarlyStopHandler(
            patience=self.config.training.early_stopping_patience, 
            min_delta=1e-4,
            score_function=lambda x: x.state.metrics[x.state.key_metric_name], 
            trainer=self
        )
        self.evaluator.add_event_handler(
            ignite.engine.Events.COMPLETED, 
            early_stopping
        )
        
    def _add_metrics_logger(self) -> None: 
        self.metric_logger=MetricLogger(
            evaluator=self.evaluator
        )
        self.metric_logger.attach(self)

    def _add_progress_bars(self) -> None:
        trainer_pbar=ProgressBar()
        evaluator_pbar=ProgressBar(
            colour='green'
        )
        trainer_pbar.attach(
            self, 
            output_transform=lambda output:{
                'loss': torch.tensor(
                    [x['loss'] for x in output]
                ).mean()
            }
        )
        evaluator_pbar.attach(self.evaluator)
        
    def _add_metrics_saver(self) -> None:
        metric_saver=MetricsSaver(
            save_dir=self.config.out_dir,
            metric_details='*',
            batch_transform=self._get_meta_dict,
            delimiter=','
        )
        metric_saver.attach(self.evaluator)
    
    def _add_eval_loss(self)->None: 
        # TODO improve by adding this to val handlers
        eval_loss_handler=ignite.metrics.Loss(
            loss_fn=self.loss_function,
            output_transform=lambda output: (
                output[0]['pred'].unsqueeze(0), # add batch dim
                output[0]['label'].argmax(0, keepdim=True).unsqueeze(0) # reverse one-hot, add batch dim
            )
        )
        eval_loss_handler.attach(self.evaluator, 'eval_loss')
        
    def _get_meta_dict(self, batch) -> list: 
        "Get dict of metadata from engine. Needed as `batch_transform`"
        image_cols=self.config.data.image_cols
        image_name=image_cols[0] if isinstance(image_cols, list) else image_cols
        key=f'{image_name}_meta_dict'
        return [item[key] for item in batch]   
        
    def load_checkpoint(self, checkpoint=None): 
        if not checkpoint: 
            # get name of last checkpoint
            checkpoint = os.path.join(
                self.config.model_dir, 
                f"network_{self.config.run_id}_key_metric={self.evaluator.state.best_metric:.4f}.pt"
            )
        self.network.load_state_dict(
            torch.load(checkpoint)
        )
        
    def run(self, try_resume_from_checkpoint=True) -> None:
        """Run training, if `try_resume_from_checkpoint` tries to
        load previous checkpoint stored at `self.config.model_dir`
        """ 

        if try_resume_from_checkpoint: 
            checkpoints = [
                os.path.join(
                    self.config.model_dir, 
                    checkpoint_name
                ) for checkpoint_name in os.listdir(
                    self.config.model_dir
                ) if self.config.run_id in checkpoint_name
            ]
            try: 
                checkpoint = sorted(checkpoints)[-1]
                self.load_checkpoint(checkpoint)
                print(f"resuming from previous checkpoint at {checkpoint}")
            except: pass # train from scratch
        
        # train the model
        super().run()
        
        # make metrics and losses more accessible
        self.loss={
            "iter": [_iter for _iter, _ in  self.metric_logger.loss],
            "loss": [_loss for _, _loss in  self.metric_logger.loss],
            "epoch": [_iter // self.state.epoch_length for _iter, _ in  self.metric_logger.loss]
            }        
        
        self.metrics={
            k: [item[1] for item in self.metric_logger.metrics[k]] for k in 
                   self.evaluator.state.metric_details.keys()
        }
       # pd.DataFrame(self.metrics).to_csv(f"{self.config.out_dir}/metric_logs.csv")
       # pd.DataFrame(self.loss).to_csv(f"{self.config.out_dir}/loss_logs.csv") 
        
    def fit_one_cycle(self, try_resume_from_checkpoint=True) -> None:
        "Run training using one-cycle-policy"
        assert "FitOneCycle" not in self.schedulers, "FitOneCycle already added"
        fit_one_cycle=monai.handlers.LrScheduleHandler(
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer, 
                max_lr=self.optimizer.param_groups[0]['lr'], 
                steps_per_epoch=self.state.epoch_length, 
                epochs=self.state.max_epochs
            ),
            epoch_level=False,
            name="FitOneCycle"
        )
        fit_one_cycle.attach(self)
        self.schedulers += ["FitOneCycle"] 
    
    def reduce_lr_on_plateau(self, 
                             try_resume_from_checkpoint=True, 
                             factor=0.1, 
                             patience=10, 
                             min_lr=1e-10, 
                             verbose=True) -> None:
        "Reduce learning rate by `factor` every `patience` epochs if kex_metric does not improve"
        assert "ReduceLROnPlateau" not in self.schedulers, "ReduceLROnPlateau already added"
        reduce_lr_on_plateau=monai.handlers.LrScheduleHandler(
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer, 
                factor=factor, 
                patience=patience, 
                min_lr=min_lr, 
                verbose=verbose
            ), 
            print_lr=True,
            name='ReduceLROnPlateau', 
            epoch_level=True, 
            step_transform=lambda engine: engine.state.metrics[engine.state.key_metric_name],
        )
        reduce_lr_on_plateau.attach(self.evaluator)
        self.schedulers += ["ReduceLROnPlateau"] 

    def evaluate(self, checkpoint=None, dataloader=None): 
        "Run evaluation with best saved checkpoint"
        self.load_checkpoint(checkpoint)
        if dataloader: 
            self.evaluator.set_data(dataloader)
            self.evaluator.state.epoch_length=len(dataloader)
        self.evaluator.run()
        print(f"metrics saved to {self.config.out_dir}") 