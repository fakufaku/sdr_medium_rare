import math
import torch
from torchiva import fftconvolve


def delete(tensor, index, dim=-1):
    """
    Removes the row at index from dimension dim
    """
    tensor = tensor.transpose(dim, -1)
    tensor = torch.cat([tensor[..., :index], tensor[..., index + 1 :]], dim=-1)
    tensor = tensor.transpose(dim, -1)
    return tensor


def mixer(audio, target_sdr, target_sir, filter_len=1):
    """
    Parameters
    ----------
    audio: numpy.ndarray, (..., n_chan, n_samples)
        The audio signals
    target_sdr: float
        The target SDR for the first channel
    target_sir: float
        The target SIR for the first channel
    filter_len: int
        The length of the filter allowed
    """

    assert target_sdr <= target_sir, "SDR has to be smaller than SIR"

    batch_shape = audio.shape[:-2]
    n_chan, n_samples = audio.shape[-2:]

    output = audio.new_zeros(audio.shape)

    # normalize power
    audio = audio / torch.std(audio, dim=-1, keepdim=True)

    for k in range(n_chan):

        random_filters = audio.new_zeros(batch_shape + (n_chan, filter_len)).normal_()
        random_filters = random_filters / math.sqrt(filter_len)
        audio_filt = fftconvolve(audio, random_filters, mode="same")

        target = audio_filt[..., k, :]
        target /= torch.std(target, dim=-1, keepdim=True)

        interf = torch.sum(delete(audio_filt, k, dim=-2), dim=-2)

        noise = audio.new_zeros(batch_shape + (n_samples,)).normal_()

        e_target = torch.var(target, dim=-1)
        e_interf = torch.var(interf, dim=-1)
        e_artif = torch.var(noise, dim=-1)

        mult_interf_sq = (e_target / e_interf) * 10 ** (-target_sir / 10)
        interf *= torch.sqrt(mult_interf_sq[..., None])
        e_interf *= mult_interf_sq

        mult_artef_sq = (e_target * 10 ** (-target_sdr / 10) - e_interf) / e_artif
        noise *= torch.sqrt(mult_artef_sq[..., None])

        output[..., k, :] = target + interf + noise

    return output
