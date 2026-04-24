# Source: https://github.com/asvspoof-challenge/asvspoof5/tree/main/Baseline-AASIST
# Original file: main.py, util.py, calculate_metrics.py, calculate_modules.py
# Note: This file is adapted from the original repository
#-----------------------------------------------------------------------------------------------
# import
import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from shutil import copy
from importlib import import_module
from typing import Dict, List, Union

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from datautils import AudioTrainDataset, AudioTestDataset, genSpoof_list, AddWhiteNoise, VolumeChange, AddFade, WaveTimeStretch, PitchShift, CodecApply, AddEnvironmentalNoise, ResampleAugmentation, AddEchoes, TimeShift, TimeMask, FreqMask, AddZeroPadding
from s_model import ConvLayers, SELayer, SERe2blocks, BiLSTM, BLDL, GraphAttentionLayer, STJGAT, Permute, AudioModel, SincConv_fast, GatingRe2blocks, WNN, FusionModel

from evaluation.calculate_metrics import calculate_minDCF_EER_CLLR
from evaluation.calculate_modules import * 
from evaluation.util import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm
#-----------------------------------------------------------------------------------------------
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Configurations
gpu = 0  # GPU id to use
torch.cuda.set_device(gpu)
#-----------------------------------------------------------------------------------------------
# Main
def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])
    dev_trial_path = (database_path /
                      "ASVspoof5.dev.track_1.tsv")
    eval_trial_path = (database_path / 
                       "ASVspoof5.eval.track_1.tsv")
    
    # define model related paths   
    selected_manipulation_key, selected_transform = augmentation(config)
    model_tag = "WavLM(V_1_F)_{}_64600_fl".format(selected_manipulation_key)
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Configurations
    gpu = 0  # GPU id to use
    torch.cuda.set_device(gpu)

    # define model architecture
    model = get_model(device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    # evaluates pretrained model 
    # NOTE: Currently it is evaluated on the development set instead of the evaluation set
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_trial_path)

        eval_dcf, eval_eer, eval_cllr = calculate_minDCF_EER_CLLR(
            cm_scores_file=eval_score_path,
            output_file=model_tag/"loaded_model_result.txt")
        print("DONE. eval_eer: {:.3f}, eval_dcf:{:.5f} , eval_cllr:{:.5f}".format(eval_eer, eval_dcf, eval_cllr))
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 100.
    best_dev_dcf = 1.
    best_dev_cllr = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("training epoch {:03d}".format(epoch))
        
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        
        score_file = metric_path / f"dev_score_{epoch}.txt"
        produce_evaluation_file(dev_loader, model, device, score_file, dev_trial_path)
        dev_eer, dev_dcf, dev_cllr = calculate_minDCF_EER_CLLR(
            cm_scores_file=score_file,
            output_file=metric_path/"dev_DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_dcf:{:.5f}, dev_cllr:{:.5f}".format(
            running_loss, dev_eer, dev_dcf, dev_cllr))
        
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_dcf", dev_dcf, epoch)
        writer.add_scalar("dev_cllr", dev_cllr, epoch)
        torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

        best_dev_dcf = min(dev_dcf, best_dev_dcf)
        best_dev_cllr = min(dev_cllr, best_dev_cllr)
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_dcf", best_dev_dcf, epoch)
        writer.add_scalar("best_dev_cllr", best_dev_cllr, epoch)
        writer.flush()
    writer.close() 
#-----------------------------------------------------------------------------------------------
# Model
def get_model(device):
    model = FusionModel(device)
    nb_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {nb_params:,}")
    return model
#-----------------------------------------------------------------------------------------------
# Data Loader
def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement"""

    audio_trn_database_path = database_path / "flac_T/"
    audio_dev_database_path = database_path / "flac_D/"
    audio_eval_database_path = database_path / "flac_E_eval/"
    trn_list_path = (database_path /
                     "ASVspoof5.train.tsv")
    dev_trial_path = (database_path /
                      "ASVspoof5.dev.track_1.tsv")
    eval_trial_path = (database_path / 
                       "ASVspoof5.eval.track_1.tsv")
    cut = 64600
    #---------------------------------------------------------------------------------------------------------------------------
    # train
    d_label_trn, file_train = genSpoof_list(
    dir_meta=trn_list_path,
    is_train=True,
    is_eval=False
    )
    print("no. train files:", len(file_train))

    train_set = AudioTrainDataset(
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=audio_trn_database_path,
        cut=cut
    )

    gen = torch.Generator()
    gen.manual_seed(seed)

    trn_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gen,
        num_workers=8
    )
    #---------------------------------------------------------------------------------------------------------------------------
    # validate
    file_dev = genSpoof_list(
        dir_meta=dev_trial_path,
        is_train=False,
        is_eval=False
    )
    print("no. dev files:", len(file_dev))

    dev_set = AudioTestDataset(
        list_IDs=file_dev,
        base_dir=audio_dev_database_path,
        cut=cut
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=8
    )
    #---------------------------------------------------------------------------------------------------------------------------
    # evaluation
    file_eval = genSpoof_list(
        dir_meta=eval_trial_path,
        is_train=False,
        is_eval=True
    )
    print("no. eval files:", len(file_eval))

    eval_set = AudioTestDataset(
        list_IDs=file_eval,
        base_dir=audio_eval_database_path,
        cut=cut
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=8
    )
    
    return trn_loader, dev_loader, eval_loader
#---------------------------------------------------------------------------------------------------------------------------------
# augmentation
def augmentation(config):
    env_noise_dataset_path = config["env_noise_dataset_path"]
    manipulations = {
        "no_augmentation": None,
        "volume_change_50": torchaudio.transforms.Vol(gain=0.5,gain_type='amplitude'),
        "volume_change_10": torchaudio.transforms.Vol(gain=0.1,gain_type='amplitude'),
        "white_noise_15": AddWhiteNoise(max_snr_db = 15, min_snr_db=15),
        "white_noise_20": AddWhiteNoise(max_snr_db = 20, min_snr_db=20),
        "white_noise_25": AddWhiteNoise(max_snr_db = 25, min_snr_db=25),
        "env_noise_wind": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="wind", env_noise_dataset_path=env_noise_dataset_path),
        "env_noise_footsteps": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="footsteps", env_noise_dataset_path=env_noise_dataset_path),
        "env_noise_breathing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="breathing", env_noise_dataset_path=env_noise_dataset_path),
        "env_noise_coughing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="coughing", env_noise_dataset_path=env_noise_dataset_path),
        "env_noise_rain": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="rain", env_noise_dataset_path=env_noise_dataset_path),
        "env_noise_clock_tick": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu",  noise_category="clock_tick", env_noise_dataset_path=env_noise_dataset_path),
        "env_noise_sneezing": AddEnvironmentalNoise(max_snr_db=20, min_snr_db=20, device="cpu", noise_category="sneezing", env_noise_dataset_path=env_noise_dataset_path),
        "pitchshift_up_110": PitchShift(max_pitch=1.10, min_pitch=1.10, bins_per_octave=12),
        "pitchshift_up_105": PitchShift(max_pitch=1.05, min_pitch=1.05, bins_per_octave=12),
        "pitchshift_down_095": PitchShift(max_pitch=0.95, min_pitch=0.95, bins_per_octave=12),
        "pitchshift_down_090": PitchShift(max_pitch=0.90, min_pitch=0.90, bins_per_octave=12),
        "timestretch_110": WaveTimeStretch(max_ratio=1.10, min_ratio=1.10, n_fft=128),
        "timestretch_105": WaveTimeStretch(max_ratio=1.05, min_ratio=1.05, n_fft=128),
        "timestretch_095": WaveTimeStretch(max_ratio=0.95, min_ratio=0.95, n_fft=128),
        "timestretch_090": WaveTimeStretch(max_ratio=0.90, min_ratio=0.90, n_fft=128),
        "echoes_1000_02": AddEchoes(max_delay=1000, max_strengh=0.2, min_delay=1000, min_strength=0.2),
        "echoes_1000_05": AddEchoes(max_delay=1000, max_strengh=0.5, min_delay=1000, min_strength=0.5),
        "echoes_2000_05": AddEchoes(max_delay=2000, max_strengh=0.5, min_delay=2000, min_strength=0.5),
        "time_shift_1600": TimeShift(max_shift=1600, min_shift=1600),
        "time_shift_16000": TimeShift(max_shift=16000, min_shift=16000),
        "time_shift_32000": TimeShift(max_shift=32000, min_shift=32000),
        "freq_mask_03": FreqMask(prob=0.3),
        "freq_mask_05": FreqMask(prob=0.5),
        "time_mask_03": TimeMask(time_mask_param=10000, p=0.3),
        "time_mask_05": TimeMask(time_mask_param=10000, p=0.5),
        "fade_50_linear": AddFade(max_fade_size=0.5,fade_shape='linear', fix_fade_size=True),
        "fade_30_linear": AddFade(max_fade_size=0.3,fade_shape='linear', fix_fade_size=True),
        "fade_10_linear": AddFade(max_fade_size=0.1,fade_shape='linear', fix_fade_size=True),
        "fade_50_exponential": AddFade(max_fade_size=0.5,fade_shape='exponential', fix_fade_size=True),
        "fade_50_quarter_sine": AddFade(max_fade_size=0.5,fade_shape='quarter_sine', fix_fade_size=True),
        "fade_50_half_sine": AddFade(max_fade_size=0.5,fade_shape='half_sine', fix_fade_size=True),
        "fade_50_logarithmic": AddFade(max_fade_size=0.5,fade_shape='logarithmic', fix_fade_size=True),
        "resample_15000": ResampleAugmentation([15000]),
        "resample_15500": ResampleAugmentation([15500]),
        "resample_16500": ResampleAugmentation([16500]),
        "resample_17000": ResampleAugmentation([17000]),
    }
    selected_manipulation_key = "no_augmentation"
    selected_transform = manipulations[selected_manipulation_key]

    return selected_manipulation_key, selected_transform
#-----------------------------------------------------------------------------------------------
# Preprocessing
def preprocessing(
    is_train,
    X_audio: torch.Tensor,
    y: torch.Tensor,
    model, encoder, criterion, optimizer, device, cut_length, config, selected_transform=None, augmentations_on_cpu=None, args=None):
    selected_manipulation_key, selected_transform = augmentation(config)
    
    X_audio = X_audio.squeeze(1).to(device)
    labels = y.to(device)
    
    if is_train:
        if augmentations_on_cpu is not None:
            X_audio = augmentations_on_cpu(X_audio)
        X_audio = X_audio.to(device)

        audio_length = X_audio.shape[-1]
        if (labels == 0).any():
            spoof_audio = X_audio[labels == 0]
            batch_size = spoof_audio.size(0)
            augmented_audio_list = []

            for i in range(batch_size):
                if selected_transform is None:
                    transformed = spoof_audio[i]
                else:
                    transformed = selected_transform(spoof_audio[i].unsqueeze(0)).squeeze(0)

                if transformed.shape[-1] > audio_length:
                    transformed = transformed[:, :audio_length]
                elif transformed.shape[-1] < audio_length:
                    transformed = transformed.repeat(1, int(audio_length/transformed.shape[-1])+1)[:, :audio_length]

                transformed = transformed.to(X_audio.device)
                augmented_audio_list.append(transformed)

            augmented_audio = torch.stack(augmented_audio_list)
            X_audio[labels == 0] = augmented_audio

    if is_train:
        return X_audio, y
#-----------------------------------------------------------------------------------------------
# Eval(validation)
def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    cut_length = 64600
    
    for X_audio, utt_id in tqdm(data_loader):
        X_audio = X_audio.to(device)
        
        with torch.no_grad():
            out_stj, out_bldl, out_lfreq, out_hfreq = model(X_audio)
            scores_stj = out_stj[:, 1]
            scores_bldl = out_bldl[:, 1]
            scores_lfreq = out_lfreq[:, 1]
            scores_hfreq = out_hfreq[:, 1]
            
            fused_scores = 0.4 * scores_stj + 0.4 * scores_bldl + 0.1 * scores_lfreq + 0.1 * scores_hfreq
            batch_score = fused_scores.data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    #assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            spk_id, utt_id, _, _, _, _, _, src, key, _ = trl.strip().split()
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(spk_id, utt_id, sco, key))
    print("Scores saved to {}".format(save_path))
#-----------------------------------------------------------------------------------------------
# Train
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = FocalLoss(weight=weight, gamma=2.0)
    
    for X_audio, y in tqdm(trn_loader):
        batch_size = X_audio.size(0)
        num_total += batch_size
        
        selected_transform = augmentation(config)
        X_audio, y = preprocessing(
            is_train=True,
            X_audio=X_audio,
            y=y,
            model=None,
            encoder=None,
            criterion=None,
            optimizer=None,
            config=config,
            device=device,
            cut_length=64600,
            selected_transform=selected_transform,
            augmentations_on_cpu=None
        )
            
        X_audio = X_audio.to(device)
        y = y.view(-1).type(torch.int64).to(device)
        
        out_stj, out_bldl, out_lfreq, out_hfreq = model(X_audio)

        loss_stj = criterion(out_stj, y)
        loss_bldl = criterion(out_bldl, y)
        loss_lfreq = criterion(out_lfreq, y)
        loss_hfreq = criterion(out_hfreq, y)

        loss = 0.25 * (loss_stj + loss_bldl + loss_lfreq + loss_hfreq)
        running_loss += loss.item() * batch_size
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss
#-----------------------------------------------------------------------------------------------
# Parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        default='./config.conf',
                        help="configuration file")
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())