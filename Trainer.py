import imp
import os
import cv2
import time
import json
import wandb
import torch
import random
import numpy as np
import seaborn as sns
import subprocess
from colorama import Fore, Style
from progress_bar import ProgressBar
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision import transforms
from pytorch_grad_cam import GradCAM

from data_utils import generate_video

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1-image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class Logger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def add(self, batch_metrics):
        if self.metrics == {}:
            for key, value in batch_metrics.items():
                self.metrics[key] = [value]
        else:
            for key, value in batch_metrics.items():
                self.metrics[key].append(value)

    def get(self):
        return {key: np.mean(value) for key, value in self.metrics.items()}

    def msg(self):
        avg_metrics = {key: np.mean(value) for key, value in self.metrics.items()}
        return avg_metrics

def print_args(args):
    print("\n---- Experiment Configuration ----")
    args_ = vars(args)
    for arg, value in args_.items():
        print(f" * {arg} => {value}")
    print("----------------------------------")

class Trainer:
    def __init__(self, model, dm, mode, args):

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.model = model
        self.mode = mode
        self.args = args
        self.vis = args.vis
        
        if self.mode == 'train':
            self.trainloader = dm.train_dataloader()
            self.valloader = dm.val_dataloader()
        elif self.mode == 'test':
            self.testloader = dm.test_dataloader()
            self.test_dataset = dm.test_dataset

        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)

        self.train_steps = 0
        self.start_epoch = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.args!= None and self.args.wandb:
            wandb.init(name=self.args.out_dir, project="DDP_IITM")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger = Logger()

        self.file_name = 'best_1.ckpt'
        
        os.makedirs(self.args.out_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.args.out_dir, self.file_name)):
            ckpt = torch.load(os.path.join(self.args.out_dir, self.file_name), map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt["optim"])
            self.lr_sched.load_state_dict(ckpt["lr_sched"])
            self.start_epoch = ckpt["epoch"]
            print(f"{Fore.RED} Loaded checkpoint. Resuming from {self.start_epoch+1} epochs...{Style.RESET_ALL}")
        else:
            self.start_epoch = 0
            print(f"{Fore.RED}Checkpoint not found. Starting fresh...{Style.RESET_ALL}")

        os.makedirs(args.out_dir, exist_ok=True)
        if os.path.exists(os.path.join(args.out_dir, f"logs_{self.args.out_dir}.txt")):
            self.log_f = open(os.path.join(args.out_dir, f"logs_{self.args.out_dir}.txt"), "a")
        else:
            self.log_f = open(os.path.join(args.out_dir, f"logs_{self.args.out_dir}.txt"), "w")

        with open(os.path.join(self.args.out_dir, f"logs_{self.args.out_dir}.txt"), "w") as f:
            json.dump(self.args.__dict__, f, indent=4)
            
    def training_step(self, epoch):
        self.logger.reset()
        self.model.train()
        
        pbar = ProgressBar(n_total=len(self.trainloader),desc='Training', epoch=epoch+1)
        for count, (inputs, target) in enumerate(self.trainloader):
            
            inputs = inputs.to(self.device).float()
            target = target.to(self.device).long()
            
            outputs = self.model(inputs)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            
            _, predicted = outputs.max(1)
            acc = predicted.eq(target.view_as(predicted)).sum().item() / inputs.shape[0]

            self.logger.add({"train loss": loss.item(), "train acc":(acc)*100})

            avg_metrics = self.logger.msg()
            pbar(step=count, info = avg_metrics)

            if self.args.wandb:
                wandb.log({"train_step": self.train_steps, "train loss": avg_metrics['train loss'], 'train_acc': avg_metrics['train acc']})
            self.train_steps += 1
    
    @torch.no_grad()
    def validation_step(self, epoch):
        self.logger.reset()
        self.model.eval()

        pbar = ProgressBar(n_total=len(self.valloader),desc='Validation', epoch=epoch+1)
        for count, (inputs, target) in enumerate(self.valloader):
            
            inputs = inputs.to(self.device).float()
            target = target.to(self.device).long()
            
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, target)
            
            _, predicted = outputs.max(1)
            acc = predicted.eq(target.view_as(predicted)).sum().item() / inputs.shape[0]

            self.logger.add({"val loss": loss.item(), "val acc":(acc)*100})

            avg_metrics = self.logger.msg()
            pbar(step=count, info = avg_metrics)

            if self.args.wandb:
                wandb.log({"val loss": avg_metrics['val loss'], 'val_acc': avg_metrics['val acc']})
            self.train_steps += 1


    def fit(self, num_epochs):
        best_valid_acc = 0

        for epoch in range(self.start_epoch, num_epochs):
            start_epoch_time = time.time()
            
            self.training_step(epoch)
            train_loss = self.logger.get()["train loss"]
            train_acc = self.logger.get()["train acc"]

            self.validation_step(epoch)            
            val_loss = self.logger.get()["val loss"]
            val_acc = self.logger.get()["val acc"]

            end_epoch_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_epoch_time, end_epoch_time)

            msg1 = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s'
            msg2 = f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}'
            msg3 = f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc:.3f}\n'

            print(msg1)
            print(msg2)
            print(msg3)

            self.log_f.write(msg1 + '\n' + msg2 + '\n' + msg3 + '\n')

            if best_valid_acc < val_acc:
                torch.save({
                    "model": self.model.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    "lr_sched": self.lr_sched.state_dict(),
                    "epoch": epoch
                }, os.path.join(self.args.out_dir, f'best_{val_acc}.ckpt'))
                print(f'\033[93mVal. Acc increased from {best_valid_acc:.3f} to {val_acc:.3f}\033[00m\n')
                best_valid_acc = val_acc

            if self.args.wandb:
                    wandb.log({"epoch": epoch, "train loss": train_loss, "train acc": train_acc, "val loss": val_loss, "val acc": val_acc, 'lr':self.lr_sched.get_lr()[0]})     

            self.lr_sched.step()
    
    @torch.no_grad()
    def predict(self):
        self.logger.reset()
        self.model.eval()
                
        pbar = ProgressBar(n_total=len(self.testloader),desc='Inference', epoch=0)
        for count, (inputs, target) in enumerate(self.testloader):
            
            inputs = inputs.to(self.device).float()
            target = target.to(self.device).long()
            
            outputs = self.model(inputs)
            
            _, predicted = outputs.max(1)
            acc = predicted.eq(target.view_as(predicted)).sum().item() / inputs.shape[0]

            # print("\nTarget : " , target)
            # print("Predicted : " , predicted)
            # print("Correct : " , correct)
            # print("Total : " , total)
            # print("Test Accuracy : " , (correct/total)*100)
            # print("Mukund Accuracy : " , acc)

            self.logger.add({"test acc":(acc)*100})

            avg_metrics = self.logger.msg()
            pbar(step=count, info = avg_metrics)

        return avg_metrics['test acc']

    def explain_model(self):
        self.logger.reset()
        self.model.eval()

        dataset = self.test_dataset
        transform = transforms.Compose([ImglistToTensor(), transforms.Resize(256), transforms.CenterCrop(224)])
        
        pbar = ProgressBar(n_total=len(dataset),desc='Inference', epoch=0)
        for count, video_data in enumerate(dataset):
            
            video_file = video_data[0]
            video_label = video_data[1]

            video_tensor = transform(video_file).unsqueeze(0).cuda().float()
            raw_imgs = tensor_to_frames(video_tensor)

            inputs = video_tensor.to(self.device).float()
            
            with torch.no_grad():
                _, w = self.model(inputs)

            spatial_attention(video_tensor, raw_imgs, self.model, count, self.args.out_dir)
            temporal_attention(raw_imgs, w, count, self.args.out_dir)
            
            avg_metrics = self.logger.msg()
            pbar(step=count, info = avg_metrics)
            

def tensor_to_frames(tensor):
    transform = T.ToPILImage()
    tensor = tensor.squeeze(0) # T x C x H x W
    img_list = []
    for t in range(tensor.size(0)):
        img_list.append(transform(tensor[t,:,:,:]))
    return img_list

class ImglistToTensor(torch.nn.Module):
    @staticmethod
    def forward(img_list):
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])

def temporal_attention(raw_imgs, w, prediction_sample_count, out_dir):

    main_out_dir = os.path.join(out_dir, f'trial_{prediction_sample_count}')
    os.makedirs(main_out_dir, exist_ok=True)
    attn_out_dir = os.path.join(out_dir, f'trial_{prediction_sample_count}', 'temporal_attention')
    os.makedirs(attn_out_dir, exist_ok=True)
    attn_map_out_dir = os.path.join(out_dir, f'trial_{prediction_sample_count}', 'attention_map')
    os.makedirs(attn_map_out_dir, exist_ok=True)
    raw_out_dir = os.path.join(out_dir, f'trial_{prediction_sample_count}', 'raw')
    os.makedirs(raw_out_dir, exist_ok=True)

    result = torch.eye(w[0].size(-1)).cuda()
    
    with torch.no_grad():
        for layer_id, attention in enumerate(w):
            #attention_heads_fused = attention.mean(axis=1)
            attention_heads_fused = attention.max(axis=1)[0]
            
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*0.7), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1)).cuda()
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    v = result[0, 0 , 1 :]

    att_mat = torch.stack(w, dim=0)
    att_mat = att_mat.squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)

    avg_att_mat = att_mat.mean(dim=0)[0,1:].detach().cpu().numpy()

    sns.heatmap([avg_att_mat])
    plt.savefig(os.path.join(main_out_dir, f'attention_heatmap'),bbox_inches='tight', pad_inches = 0)
    plt.clf()

    for i in range(len(raw_imgs)):
        temp = (raw_imgs[i])
        plt.imshow(temp)
        plt.axis('off')
        plt.savefig(os.path.join(raw_out_dir, f'frame_{i}'),bbox_inches='tight', pad_inches = 0)
        plt.clf()
        
    mask = v.detach().cpu().numpy()
    mask = mask / mask.max()
    #print(mask)
    for i in range(len(mask)):
        temp = (mask[i] * raw_imgs[i])
        plt.imshow(temp.astype("uint8"))
        plt.axis('off')
        plt.savefig(os.path.join(attn_out_dir, f'frame_{i}'),bbox_inches='tight', pad_inches = 0)
        plt.clf()
        sns.heatmap(np.full((224, 224), mask[i]), vmin=0, vmax=1, cmap='viridis', cbar=False)
        plt.axis('off')
        plt.savefig(os.path.join(attn_map_out_dir, f'frame_{i}'),bbox_inches='tight', pad_inches = 0)
        plt.clf()

    generate_video.generate(os.path.join(out_dir, f'trial_{prediction_sample_count}', 'temporal_attention'))
    generate_video.generate(os.path.join(out_dir, f'trial_{prediction_sample_count}', 'attention_map'))
    generate_video.generate(os.path.join(out_dir, f'trial_{prediction_sample_count}', 'spatial_attention'))
    generate_video.generate(os.path.join(out_dir, f'trial_{prediction_sample_count}', 'raw'))
    file1 = os.path.join(out_dir, f'trial_{prediction_sample_count}', 'attention_map', 'all.jpg')
    file2 = os.path.join(out_dir, f'trial_{prediction_sample_count}', 'raw', 'all.jpg')
    file3 = os.path.join(out_dir, f'trial_{prediction_sample_count}', 'temporal_attention', 'all.jpg')
    file4 = os.path.join(out_dir, f'trial_{prediction_sample_count}', 'spatial_attention', 'all.jpg')
    generate_video.vertical_stack(file1, file2, file3, file4, os.path.join(out_dir, f'trial_{prediction_sample_count}'))

def spatial_attention(video_tensor, raw_imgs, model, prediction_sample_count, out_dir):

    attn_out_dir = os.path.join(out_dir, f'trial_{prediction_sample_count}', 'spatial_attention')
    os.makedirs(attn_out_dir, exist_ok=True)
    raw_out_dir = os.path.join(out_dir, f'trial_{prediction_sample_count}', 'raw')
    os.makedirs(raw_out_dir, exist_ok=True)

    resnet_module = model.encoder.resnet
    target_layers = [resnet_module.layer4[-1]]

    cam = GradCAM(model=resnet_module, target_layers=target_layers, use_cuda=True)

    for t in range(video_tensor.size(1)):
        img_tensor = video_tensor[:,t,:,:,:]

        grayscale_cam = cam(input_tensor=img_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(np.array(raw_imgs[t])/255, grayscale_cam, use_rgb=True, image_weight=0.7)
        plt.imshow(visualization)
        plt.axis('off')
        plt.savefig(os.path.join(attn_out_dir, f'frame_{t}'),bbox_inches='tight', pad_inches = 0)
        plt.clf()