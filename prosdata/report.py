import os
import io
import cv2
import tqdm
import torch
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ReportGenerator():
    "Generate markdown document, summarizing the training"
    
    def __init__(self, run_id, out_dir=None, log_dir=None):
    
        self.run_id, self.out_dir, self.log_dir = run_id, out_dir, log_dir

        if log_dir: 
            self.train_logs = pd.read_csv(os.path.join(log_dir, 'train_logs.csv'))
            self.metric_logs = pd.read_csv(os.path.join(log_dir, 'metric_logs.csv'))
        if out_dir: 
            self.dice = pd.read_csv(os.path.join(out_dir, 'MeanDice_raw.csv'))
            self.hausdorf = pd.read_csv(os.path.join(out_dir, 'HausdorffDistance_raw.csv'))
            self.surface = pd.read_csv(os.path.join(out_dir, 'SurfaceDistance_raw.csv'))

            self.mean_metrics = pd.DataFrame(
                {"mean_dice" : [round(np.mean(self.dice[col]),3) for col in self.dice if col.startswith('class')],  
                 "mean_hausdorf" : [round(np.mean(self.hausdorf[col]),3) for col in self.hausdorf if col.startswith('class')],   
                 "mean_surface" : [round(np.mean(self.surface[col]),3) for col in self.surface if col.startswith('class')] 
                }).transpose()
    
    def generate_report(self, loss_plot=True, metric_plot=True, boxplots=True, animation=True):
        fn = os.path.join(self.run_id, 'report', 'SegmentationReport.md')
        os.makedirs(os.path.join(self.run_id, 'report'), exist_ok=True)
        with open(fn, 'w+') as f: 
            f.write('# Segmentation Report\n\n')
        
        if loss_plot: 
            fig = self.plot_loss(self.train_logs, self.metric_logs)
            plt.savefig(os.path.join(self.run_id, 'report', 'loss_and_lr.png'), dpi = 150)

            with open(fn, 'a') as f: 
                f.write('## Loss, LR-Schedule and Key Metric\n')
                f.write('![Loss, LR-Schedule and Key Metric](loss_and_lr.png)\n\n')

        if metric_plot: 
            fig = plt.figure("metrics", (18, 6))

            ax = plt.subplot(1, 3, 1)
            plt.ylim([0,1])
            plt.title("Mean Dice")
            plt.xlabel("epoch")
            plt.plot(self.metric_logs.index, self.metric_logs.MeanDice)

            ax = plt.subplot(1, 3, 2)
            plt.title("Mean Hausdorff Distance")
            plt.xlabel("epoch")
            plt.plot(self.metric_logs.index, self.metric_logs.HausdorffDistance)

            ax = plt.subplot(1, 3, 3)
            plt.title("Mean Surface Distance")
            plt.xlabel("epoch")
            plt.plot(self.metric_logs.index, self.metric_logs.SurfaceDistance)

            plt.savefig(os.path.join(self.run_id, 'report', 'metrics.png'), dpi = 150)
            fig.clear()
            plt.close()

            with open(fn, 'a') as f: 
                f.write('## Metrics\n')
                f.write('![metrics](metrics.png)\n\n')

        if boxplots: 
            fig = plt.figure("boxplots", (18, 6))

            ax = plt.subplot(1, 3, 1)
            plt.title("Dice")
            plt.xlabel("class")
            plt.boxplot(self.dice[[col for col in self.dice if col.startswith('class')]])

            ax = plt.subplot(1, 3, 2)
            plt.title("Hausdorff Distance")
            plt.xlabel("class")
            plt.boxplot(self.hausdorf[[col for col in self.hausdorf if col.startswith('class')]])

            ax = plt.subplot(1, 3, 3)
            plt.title("Surface Distance")
            plt.xlabel("class")
            plt.boxplot(self.surface[[col for col in self.surface if col.startswith('class')]])

            plt.savefig(os.path.join(self.run_id, 'report', 'boxplots.png'),dpi = 150)

            fig.clear()
            plt.close()

            with open(fn, 'a') as f: 
                f.write(f"## Individual metrics\n\n")
                f.write(f"{self.mean_metrics.to_markdown()}\n\n")
                f.write(f"![boxplot](boxplots.png)\n\n")
        if animation: 
            self.generate_gif()
            with open(fn, 'a') as f: 
                f.write('## Visualization of progress\n')
                f.write('![progress](progress.gif)\n\n')

    def plot_loss(self, train_logs, metric_logs): 
        iteration = train_logs.iteration/sum(train_logs.epoch == 1)
        fig = plt.figure("loss and lr", (12, 6))

        y_max = max(metric_logs.eval_loss) + 0.5
        if y_max > 3: y_max = 3

        ax = plt.subplot(1, 2, 1)
        plt.ylim([0,y_max])
        plt.title("Epoch Average Loss")
        plt.xlabel("epoch")
        plt.plot(iteration, train_logs.loss)
        plt.plot(metric_logs.index, metric_logs.eval_loss)

        ax = plt.subplot(1, 2, 2)
        ax.set_yscale('log')
        plt.title("LR Schedule")
        plt.xlabel("epoch")
        plt.plot(iteration, train_logs.lr)
        return fig

    def get_arr_from_fig(self, fig, dpi=180):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def get_slices(self, im, slices): 
        ims = torch.unbind(im[:, :, slices], -1) # extract n slices
        ims = [i.transpose(0,1).flip(0) for i in ims] # rotate slices 90 degrees
        if len(slices) > 4 and len(slices) % 2 == 0: 
            n = len(slices) // 2
            ims1 = torch.cat(ims[0:n], 1)
            ims2 = torch.cat(ims[n:], 1)
            return torch.cat([ims1, ims2], 0)
        else: 
            return torch.cat(ims, 1) # create tile
        
    def plot_images(self, fns, slices, cmap='Greys_r', figsize=15, **kwargs): 
        ims = [torch.load(os.path.join(self.out_dir, 'preds', fn)).cpu().argmax(0) for fn in fns]
        ims = [self.get_slices(im, slices) for im in ims]
        ims = torch.cat(ims, 0)
        plt.figure(figsize=(figsize,figsize)) 
        plt.imshow(ims, cmap=cmap, **kwargs)
        plt.axis('off')
        
    def load_segmentation_image(self, fn): 
        im = torch.load(fn).cpu().unsqueeze(0)
        im = torch.nn.functional.interpolate(im, (224, 224, 112))
        im = im.argmax(1).squeeze()
        im = self.get_slices(im, slices = (40, 48, 56, 74, 82, 90)) 
        im = im/im.max() * 255
        return im
    
    def generate_gif(self):
        with imageio.get_writer(
            os.path.join(self.run_id,'report','progress.gif'),
            mode='I', 
            fps = max(self.train_logs.epoch) // 10) as writer: # make gif 10 seconds
            for epoch in tqdm.tqdm(list(self.train_logs.epoch.unique())):
                seg_fn = os.path.join(self.out_dir, 'preds', f"pred_epoch_{epoch}.pt")
                if os.path.exists(seg_fn): im = self.load_segmentation_image(seg_fn)

                plt_train_logs = self.train_logs[self.train_logs.epoch <= epoch]
                loss_plt = self.plot_loss(plt_train_logs, self.metric_logs[:epoch])
                loss_fig = self.get_arr_from_fig(loss_plt)[:,:,0]

                new_shape = im.shape[1], int(loss_fig.shape[0] / loss_fig.shape[1] * im.shape[1])
                loss_fig = cv2.resize(loss_fig, (im.shape[1], im.shape[0]))

                images = torch.cat([im, torch.tensor(loss_fig)], 0).numpy().astype(np.uint8)  
                writer.append_data(images)

                loss_plt.clear()
                plt.close()