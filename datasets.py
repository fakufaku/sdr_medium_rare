import json
import math
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile

import torchiva


def compute_interval(n_target, n_mix, n_originals, n_offsets):
    """
    Compute an interval containing all the sources

    Parameters
    ----------
    n_target:
        target number samples
    n_mix:
        number of samples in mixture
    n_originals
        number of samples of the sources
    n_offsets
        offset of the sources in the mixtures
    """

    if n_target >= n_mix:
        return 0, n_mix

    n_mix = np.array(n_mix)
    n_originals = np.array(n_originals)
    n_offsets = np.array(n_offsets)

    # left is the sample at which starts the source that starts last
    # right is thee sample at which terminates the first source to terminate
    left, right = np.max(n_offsets), np.min(n_offsets + n_originals)

    # the midpoint between left and right should be the center of the
    # target interval
    midpoint = 0.5 * (left + right)

    # the start and end of interval
    start = midpoint - n_target // 2
    end = start + n_target

    # handle border effects
    if start < 0:
        return 0, n_target
    elif end >= n_mix:
        return n_mix - n_target, n_mix
    else:
        return int(start), int(end)


class WSJ1SpatialDataset(torch.utils.data.Dataset):
    """
    Dataloader for the WSJ1-spatialized datasets for multichannel

    Parameters
    ----------
    metafilename: pathlib.Path or str
        the path to the mixinfo.json file containing the dataset metadata
    max_len_s: float, optional
        the length in seconds of the samples
    ref_mic: int optional
        the microphone to use for the scaling reference, if not provided
        it is chosen at random
    shuffle_channels: bool, optional
        if set to True (default), the channels of the microphones will be
        shuffled at random
    ref_is_reverb: bool, optional
        if set to True (default), the reverberant clean signal is used
        as a reference, if False, the anechoic clean signal is used
    noiseless: bool, optional
        if set to False (default), use the noisy mixture, if true, use the
        noiseless mixture
    max_n_samples: int, optional
    """

    def __init__(
        self,
        dataset_location: Union[Path, str],
        max_len_s: Optional[float] = None,
        ref_mic: Optional[int] = 0,
        shuffle_channels: Optional[bool] = False,
        ref_is_reverb: Optional[bool] = True,
        noiseless: Optional[bool] = False,
        max_n_samples: Optional[int] = None,
        filter_dc: Optional[bool] = False,
        remove_mean: Optional[bool] = False,
    ):
        super().__init__()

        self.dataset_location = Path(dataset_location)
        self.metafilename = self.dataset_location / "mixinfo_noise.json"
        self.max_len_s = max_len_s
        self.ref_mic = ref_mic
        self.shuffle_channels = shuffle_channels
        self.ref_is_reverb = ref_is_reverb
        self.noiseless = noiseless
        self.filter_dc = filter_dc
        self.remove_mean = remove_mean

        # open the metadata and find the dataset path
        with open(self.metafilename, "r") as f:
            # the metadata is stored as a dict, but a list is preferable
            self.metadata = list(json.load(f).values())

        # we truncate the dataset if required
        if max_n_samples is not None and max_n_samples != -1:
            self.metadata = self.metadata[:max_n_samples]

    def __len__(self):
        return len(self.metadata)

    def get_mixinfo(self, idx):
        return self.metadata[idx]

    def __getitem__(self, idx):

        room = self.metadata[idx]

        if self.noiseless:
            mix_fn = Path(room["wav_dpath_mixed_reverberant"])
        else:
            mix_fn = Path(room["wav_mixed_noise_reverb"])

        # hack to get relative path from dataset directory
        mix_fn = Path("").joinpath(*mix_fn.parts[-3:])

        if self.ref_is_reverb:
            ref_fns_list = room["wav_dpath_image_reverberant"]
        else:
            ref_fns_list = room["wav_dpath_image_anechoic"]
        ref_fns = [Path(p) for p in ref_fns_list]

        # hack to get relative path from dataset directory
        ref_fns = [Path("").joinpath(*fn.parts[-3:]) for fn in ref_fns]

        # load the mixture audio
        # torchaudio loads the data, converts to float, and normalize to [-1, 1] range
        audio_mix, fs_1 = torchaudio.load(self.dataset_location / mix_fn)

        # now we know the number of channels
        n_channels = audio_mix.shape[0]

        # randomly shuffle the order of the channels in the mixture if required
        if self.shuffle_channels:
            p = torch.randperm(n_channels)
        else:
            p = torch.arange(n_channels)

        # the reference mic needs to be picked according the shuffled order
        ref_mic = p[self.ref_mic]

        # now load the references
        audio_ref_list = []
        for fn in ref_fns:
            audio, fs_2 = torchaudio.load(self.dataset_location / fn)

            assert fs_1 == fs_2

            audio_ref_list.append(audio[ref_mic, None, :])

        audio_ref = torch.cat(audio_ref_list, dim=0)

        audio_mix = audio_mix[p]

        if self.remove_mean:
            audio_mix = audio_mix - audio_mix.mean(dim=-1, keepdim=True)
            audio_ref = audio_ref - audio_ref.mean(dim=-1, keepdim=True)

        if self.filter_dc:
            audio_mix = torchiva.filter_dc(audio_mix)
            audio_ref = torchiva.filter_dc(aufio_ref)

        if self.max_len_s is None:
            return audio_mix, audio_ref
        else:
            # the length of the different signals
            n_target = int(fs_1 * self.max_len_s)
            n_originals = room["wav_n_samples_original"]
            n_offsets = room["wav_offset"]
            n_mix = audio_mix.shape[-1]

            # compute an interval that has all sources in it
            s, e = compute_interval(n_target, n_mix, n_originals, n_offsets)

            return (audio_mix[..., s:e], audio_ref[..., s:e])


def collator(batch_list):
    """
    Collate a bunch of multichannel signals based
    on the size of the longest sample. The samples are cut at the center
    """
    max_len = max([s[0].shape[-1] for s in batch_list])

    n_batch = len(batch_list)
    n_channels_data = batch_list[0][0].shape[0]
    n_channels_target = batch_list[0][1].shape[0]

    data = batch_list[0][0].new_zeros((n_batch, n_channels_data, max_len))
    target = batch_list[0][1].new_zeros((n_batch, n_channels_target, max_len))

    offsets = [(max_len - s[0].shape[-1]) // 2 for s in batch_list]

    for b, ((d, t), o) in enumerate(zip(batch_list, offsets)):
        data[b, :, o : o + d.shape[-1]] = d
        target[b, :, o : o + t.shape[-1]] = t

    # handle when extra things are provided (e.g. transcripts)
    rest_of_items = []
    for i in range(len(batch_list[0]) - 2):
        rest_of_items.append([b[i + 2] for b in batch_list])

    return data, target, *rest_of_items


class InterleavedLoaders:
    """
    This is a wrapper to sample alternatively from several dataloaders

    It samples until all of them are exhausted

    Parameters
    ----------
    dataloaders: list of torch.utils.data.DataLoader
        A list that contains all the dataloaders we want to sample from
    """

    def __init__(self, dataloaders: List[torch.utils.data.DataLoader]):
        self.dataloaders = dataloaders
        self._reset_dataloader_iter()

    def _reset_dataloader_iter(self):
        self.dataloaders_iter = [iter(d) for d in self.dataloaders]

    def __len__(self):
        return sum([len(dl) for dl in self.dataloaders])

    def __iter__(self):
        return self

    def __next__(self):
        for i, dl in enumerate(self.dataloaders_iter):
            try:
                # we loop until we find a non-empty loader
                batch = next(dl)
                # then we put dataloader at the end of the list
                self.dataloaders_iter = (
                    self.dataloaders_iter[i + 1 :] + self.dataloaders_iter[: i + 1]
                )
                # return the batch
                return batch
            except StopIteration:
                continue

        self._reset_dataloader_iter()

        raise StopIteration


def reverb2mix_transcript_parse(path):
    """
    Parse the file format of the MLF files that
    contains the transcripts in the REVERB challenge
    dataset
    """

    utterances = {}

    with open(path, "r") as f:
        everything = f.read()

    all_utt = everything.split("\n.\n")

    for i, utt in enumerate(all_utt):

        if i == 0:
            assert utt[:7] == "#!MLF!#"
            utt = utt[7:]

        words = utt.split("\n")

        label = words[0][4:-6]
        sentence = " ".join(words[1:])

        speaker = label[:-5]
        utterance = label[-5:]

        utterances[label] = {
            "utterance_id": utterance,
            "speaker_id": speaker,
            "transcript": sentence,
        }

    return utterances


class REVERB2MIX(torch.utils.data.Dataset):
    def __init__(
        self,
        reverb_path,
        reverb2mix_path,
        data_type="real",
        distance="near",
        room_id=1,
        n_channels=8,
        ref_mic=0,
    ):

        self.basepath_reverb2mix = Path(reverb2mix_path)
        self.basepath_reverb = Path(reverb_path)

        assert distance in ["near", "far", None], "distance shoulde near or far"
        if distance is None:
            self.distances = ["near", "far"]
        else:
            self.distances = [distance]

        assert 1 <= n_channels <= 8, "number of channels is between 1 and 8"
        self.n_channels = n_channels

        assert 0 <= ref_mic < self.n_channels
        self.ref_mic = ref_mic

        self.fs = 16000  # fixed in dataset

        if data_type == "real":
            self.mixinfo = {}  # store all the info in this dict

            assert room_id == 1, "For data_type=real, only room_id=1 is available"
            data_type_tag = "RealData"

            for distance in self.distances:

                mixfilename = f"RealData_et_for_8ch_{distance}_room{room_id}_wav.scp"
                self.mixfile_path = (
                    self.basepath_reverb2mix / f"REVERB_2MIX/scps/{mixfilename}"
                )

                with open(self.mixfile_path, "r") as f:
                    for line in f.readlines():
                        label, wavpath, *_ = line.strip().split(" ")
                        self.mixinfo[label] = {
                            "data_id": label,
                            "wav_path_mix": str(
                                self.basepath_reverb2mix / f"REVERB_2MIX/{wavpath}"
                            ),
                            "wav_path_images": [[], []],  # there are always 2 sources
                            "transcript_espnet": [],
                            "speaker_id": [],
                            "utterance_id": [],
                            "distance": distance,
                            "wav_frame_rate_mixed": self.fs,
                        }

                # now get all the paths for the original sources
                for isrc, src in enumerate(["et", "dt"]):
                    for ch in range(1, 8 + 1):
                        fn = f"{data_type}_{distance}_room{room_id}_{src}_ch{ch}.scp"
                        path = (
                            self.basepath_reverb2mix / f"scps_for_genreal/rev2mix/{fn}"
                        )

                        with open(path, "r") as f:
                            for line in f.readlines():
                                label, wavpath, *_ = line.strip().split(" ")
                                mixid = f"room{room_id}_{distance}_{label}"

                                self.mixinfo[mixid]["wav_path_images"][isrc].append(
                                    str(self.basepath_reverb / wavpath)
                                )

            # now we get all the transcripts
            transcripts_path = {
                "et": self.basepath_reverb / "MC_WSJ_AV_Eval/mlf/WSJ.mlf",  # Eval
                "dt": self.basepath_reverb / "MC_WSJ_AV_Dev/mlf/WSJ.mlf",  # Dev
            }

            transcripts = {}
            for s, path in transcripts_path.items():
                transcripts[s] = reverb2mix_transcript_parse(path)

            for mixid in self.mixinfo.keys():
                for isrc, src in enumerate(["et", "dt"]):
                    img_path = Path(self.mixinfo[mixid]["wav_path_images"][isrc][0])
                    fn = img_path.with_suffix("").name
                    trans_lbl = fn.split("_")[-1]
                    self.mixinfo[mixid]["transcript_espnet"].append(
                        transcripts[src][trans_lbl]["transcript"]
                    )
                    self.mixinfo[mixid]["utterance_id"].append(
                        transcripts[src][trans_lbl]["utterance_id"]
                    )
                    self.mixinfo[mixid]["speaker_id"].append(
                        transcripts[src][trans_lbl]["speaker_id"]
                    )

        elif data_type == "sim":
            raise NotImplementedError(
                "Sorry, dataset is not implemented for simulated signals"
            )
            assert room_id in [
                1,
                2,
                3,
            ], "For data_type=sim, only room_id=1, 2, or 3 are available"
            data_type_tag = "SimData"
        else:
            raise ValueError("Supported values for data_type are real or sim")

        self.mixinfo_keys = list(self.mixinfo.keys())

    def get_mixinfo(self, idx):
        return self.mixinfo[self.mixinfo_keys[idx]]

    def __len__(self):
        return len(self.mixinfo_keys)

    def __getitem__(self, idx):
        key = self.mixinfo_keys[idx]
        info = self.mixinfo[key]

        # load the mixture audio
        # torchaudio loads the data, converts to float, and normalize to [-1, 1] range
        audio_mix, fs_1 = torchaudio.load(info["wav_path_mix"])
        assert self.fs == fs_1

        # load the reference audio
        audio_ref = []
        for images_path in info["wav_path_images"]:
            aref, fs_2 = torchaudio.load(images_path[self.ref_mic])
            assert self.fs == fs_2
            audio_ref.append(aref)

        # adjust length to that of the first source
        L = audio_ref[0].shape[1]
        if audio_ref[1].shape[1] < L:
            ref2 = audio_ref[1]
            audio_ref[1] = torch.cat(
                [ref2, ref2.new_zeros((1, L - ref2.shape[1]))], dim=1
            )
        elif audio_ref[1].shape[1] > L:
            audio_ref[1] = audio_ref[1][:, :L]
        audio_ref = torch.cat(audio_ref, dim=0)

        return audio_mix, audio_ref
