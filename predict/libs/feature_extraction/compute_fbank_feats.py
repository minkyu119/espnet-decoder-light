import numpy as np
from scipy.io.wavfile import read
from copy import deepcopy
import argparse
import time
import sys
import math
from scipy.fft import rfft
import matplotlib.pyplot as plt

KALDI_COMPLEXFFT_BLOCKSIZE = 8192

def vector(size, value_type):
    return [value_type() for i in range(size)]

def matrix(row_size, col_size, value_type):
    return [[value_type() for j in range(col_size)] for i in range(row_size)]

class Pair():
    def __init__(self, first=None, second=None):
        self.first = first
        self.second = second

class LinearResample():
    def __init__(self, samp_rate_in_hz, samp_rate_out_hz, filter_cutoff_hz, num_zeros):
        self.__decl_parameters()
        self.sampling_rate_in = int(samp_rate_in_hz)
        self.sampling_rate_out = int(samp_rate_out_hz)
        self.filter_cutoff = filter_cutoff_hz
        self.num_zeros = num_zeros

        assert samp_rate_in_hz > 0 and samp_rate_out_hz > 0 and filter_cutoff_hz > 0 and \
               filter_cutoff_hz * 2 <= samp_rate_in_hz and filter_cutoff_hz * 2 <= samp_rate_out_hz and \
               num_zeros > 0

        base_frequency = np.gcd(self.sampling_rate_in, self.sampling_rate_out)
        self.input_samples_in_unit = int(self.sampling_rate_in / base_frequency)
        self.output_samples_in_unit = int(self.sampling_rate_out / base_frequency)

        self.__set_indexes_and_weights()
        self.reset()

    def __decl_parameters(self):
        self.sampling_rate_in = int()
        self.sampling_rate_out = int()
        self.filter_cutoff = float()
        self.num_zeros = int()
        self.input_samples_in_unit = int()
        self.output_samples_in_unit = int()
        self.first_index = list()
        self.weights = list()
        self.input_sample_offset = int()
        self.output_sample_offset = int()
        self.input_remainder = list()


    def resample(self, input, flush):
        input_dim = len(input)
        total_input_sample = self.input_sample_offset + input_dim
        total_output_sample = self.__get_num_output_samples(total_input_sample, flush)

        assert total_output_sample >= self.output_sample_offset
        output = np.zeros(total_output_sample - self.output_sample_offset)

        for sample_out in range(self.output_sample_offset, total_output_sample):
            first_sample_in, sample_out_wrapped = self.__get_indexes(sample_out)
            weights = self.weights[sample_out_wrapped]

            first_input_index = int(first_sample_in - self.input_sample_offset)
            this_output = float()
            if first_input_index >= 0 and first_input_index + len(weights) <= input_dim:
                this_output = np.dot(input[first_input_index:first_input_index+len(weights)], weights)
            else:
                for i in range(len(weights)):
                    weight = weights[i]
                    input_index = first_input_index + i
                    if input_index < 0 and len(self.input_remainder) + input_index >= 0:
                        this_output += weight * self.input_remainder[len(self.input_remainder)+input_index]
                    elif input_index >= 0 and input_index < input_dim:
                        this_output += weight * input[input_index]
                    elif input_index >= input_dim:
                        assert flush

            output_index = int(sample_out - self.output_sample_offset)
            output[output_index] = this_output

        if flush:
            self.reset()
        else:
            self.__set_remainder(input)
            self.input_sample_offset = total_input_sample
            self.output_sample_offset = total_output_sample

        return output

    def reset(self):
        self.input_sample_offset = 0
        self.output_sample_offset = 0
        self.input_remainder = []
        return

    def get_input_sampling_rate(self):
        return self.sampling_rate_in

    def get_output_sampling_rate(self):
        return self.sampling_rate_out

    def __get_num_output_samples(self, input_num_sample, flush):
        tick_frequency = int(np.lcm(self.sampling_rate_in, self.sampling_rate_out))
        ticks_per_input_period = int(tick_frequency / self.sampling_rate_in)

        interval_length_in_ticks = input_num_sample * ticks_per_input_period
        if not flush:
            window_width = self.num_zeros / (2.0 * self.filter_cutoff)
            window_width_ticks = int(np.floor(window_width * tick_frequency))
            interval_length_in_ticks -= window_width_ticks

        if interval_length_in_ticks <= 0:
            return 0

        ticks_per_output_period = int(tick_frequency / self.sampling_rate_out)
        last_output_sample = int(interval_length_in_ticks / ticks_per_output_period)

        if last_output_sample * ticks_per_output_period == interval_length_in_ticks:
            last_output_sample -= 1

        num_output_sample = last_output_sample + 1

        return num_output_sample

    def __get_indexes(self, sample_out):
        unit_index = int(sample_out / self.output_samples_in_unit)
        sample_out_wrapped = int(sample_out - unit_index * self.output_samples_in_unit)
        first_sample_in = self.first_index[sample_out_wrapped] + unit_index * self.input_samples_in_unit

        return first_sample_in, sample_out_wrapped

    def __set_remainder(self, input):
        old_remainder = deepcopy(self.input_remainder)
        max_remainder_needed = int(np.ceil(self.sampling_rate_in * self.num_zeros / self.filter_cutoff))
        self.input_remainder = vector(max_remainder_needed, float)
        for index in range(-len(self.input_remainder), 0):
            input_index = index + len(input)
            if input_index >= 0:
                self.input_remainder[index + len(self.input_remainder)] = input[input_index]
            elif input_index + len(old_remainder) >= 0:
                self.input_remainder[index + len(self.input_remainder)] = old_remainder[input_index + len(old_remainder)]

        return

    def __set_indexes_and_weights(self):
        self.first_index = vector(self.output_samples_in_unit, int)
        self.weights = vector(self.output_samples_in_unit, list)

        window_width = self.num_zeros / (2.0 * self.filter_cutoff)

        for i in range(self.output_samples_in_unit):
            output_t = i / self.sampling_rate_out
            min_t = output_t - window_width
            max_t = output_t + window_width

            min_input_index = int(np.ceil(min_t * self.sampling_rate_in))
            max_input_index = int(np.floor(max_t * self.sampling_rate_in))
            num_indices = max_input_index - min_input_index + 1

            self.first_index[i] = min_input_index
            self.weights[i] = vector(num_indices, float)

            for j in range(num_indices):
                input_index = int(min_input_index + j)
                input_t = input_index / self.sampling_rate_in
                delta_t = input_t - output_t
                self.weights[i][j] = self.__filter_func(delta_t) / self.sampling_rate_in

        return

    def __filter_func(self, t):
        window = float()
        filter = float()

        if np.abs(t) < self.num_zeros / (2.0 * self.filter_cutoff):
            window = 0.5 * (1 + np.cos(2 * np.pi * self.filter_cutoff / self.num_zeros * t))
        else:
            window = 0.0

        if t != 0:
            filter = np.sin(2 * np.pi * self.filter_cutoff * t) / (np.pi * t)
        else:
            filter = 2 * self.filter_cutoff
        return filter * window

def resample_waveform(original_freq, wave, new_freq):
    min_freq = min(original_freq, new_freq)
    lowpass_cutoff = 0.99 * 0.5 * min_freq
    lowpass_filter_width = 6
    resampler = LinearResample(original_freq, new_freq, lowpass_cutoff, lowpass_filter_width)
    return resampler.resample(wave, True)

def round_up_to_nearest_power_of_two(n):
    n = int(n)
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n+1

def feature_window_function(opts):
    frame_length = opts.window_size()
    assert frame_length > 0
    a = 2 * np.pi / (frame_length - 1)
    i_range = np.arange(frame_length)
    window = None
    if opts.window_type == "hanning":
        window = 0.5 - 0.5 * np.cos(a * i_range)
    elif opts.window_type == "sine":
        window = np.sin(0.5 * a * i_range)
    elif opts.window_type == "hamming":
        window = 0.54 - 0.46 * np.cos(a * i_range)
    elif opts.window_type == "povey":
        window = np.power((0.5 - 0.5 * np.cos(a * i_range)), 0.85)
    elif opts.window_type == "rectangular":
        window = np.ones(frame_length)
    elif opts.window_type == "blackman":
        window = opts.blackman_coeff - 0.5 * np.cos(a * i_range) + \
                    (0.5 - opts.blackman_coeff) * np.cos(2 * a * i_range)
    else:
        raise Exception(f"Invalid window type {opts.window_type}")

    return window

def first_sample_of_frame(frame, opts):
    frame_shift = opts.window_shift()
    if opts.snip_edges:
        return frame * frame_shift
    else:
        midpoint_of_frame = frame_shift * frame + frame_shift / 2
        beginning_of_frame = midpoint_of_frame - opts.window_size() / 2
        return beginning_of_frame

def dither(waveform, dither_value):
    if dither_value == 0.0:
        return
    dim = len(waveform)
    rand_values = np.random.normal(size=dim)
    waveform += rand_values * dither_value
    return waveform

def preemphasize(waveform, preemph_coeff):
    from scipy.signal import lfilter
    if preemph_coeff == 0.0:
        return
    assert preemph_coeff >= 0.0 and preemph_coeff <= 1.0
    return lfilter([1, -preemph_coeff], [1], waveform)

def process_window(opts, window_option, window, log_energy_pre_window):
    frame_length = opts.window_size()
    assert len(window) == frame_length

    if opts.dither != 0.0:
        window = dither(window, opts.dither)

    if log_energy_pre_window != None:
        energy = max(np.dot(window, window), np.finfo(np.float32).eps)
        log_energy_pre_window = np.log(energy)

    if opts.preemph_coeff != 0.0:
        window = preemphasize(window, opts.preemph_coeff)

    window *= feature_window_function(window_option)
    return window, log_energy_pre_window

def extract_window(sample_offset, wave, f, opts, window_option, log_energy_pre_window):
    assert sample_offset >= 0 and len(wave) != 0
    frame_length = opts.window_size()
    frame_length_padded = opts.padded_window_size()

    num_samples = sample_offset + len(wave)
    start_sample = first_sample_of_frame(f, opts)
    end_sample = start_sample + frame_length

    if opts.snip_edges:
        assert start_sample >= sample_offset and end_sample <= num_samples
    else:
        assert sample_offset == 0 or start_sample >= sample_offset

    window = np.zeros(frame_length_padded)
    wave_start = start_sample - sample_offset
    wave_end = wave_start + frame_length

    if wave_start >= 0 and wave_end <= len(wave):
        window[:frame_length] = wave[wave_start : wave_start + frame_length]
    else:
        wave_dim = len(wave)
        for s in range(frame_length):
            s_in_wave = s + wave_start
            while s_in_wave < 0 or s_in_wave >= wave_dim:
                if s_in_wave < 0:
                    s_in_wave = -s_in_wave - 1
                else:
                    s_in_wave = 2 * wave_dim - 1 - s_in_wave
            window[s] = wave[s_in_wave]

    frame = window[:frame_length]
    window[:frame_length], log_energy_pre_window = process_window(opts, window_option, frame, log_energy_pre_window)
    return window, log_energy_pre_window

def num_frames(num_samples, opts, flush=True):
    frame_shift = opts.window_shift()
    frame_length = opts.window_size()

    if opts.snip_edges:
        if num_samples < frame_length:
            return 0
        else:
            return int(1 + ((num_samples - frame_length) / frame_shift))
    else:
        num_frames = (num_samples + (frame_shift / 2)) / frame_shift
        if flush:
            return int(num_frames)
        end_sample_of_last_frame = first_sample_of_frame(num_frames - 1, opts) + frame_length
        while num_frames > 0 and end_sample_of_last_frame > num_samples:
            num_frames -= 1
            end_sample_of_last_frame -= frame_shift
        return int(num_frames)
class DefaultOptions():
    def __init__(self):
        self.allow_downsample=False
        self.allow_upsample=False
        self.blackman_coeff=0.42
        self.channel=-1
        self.debug_mel=False
        self.dither=1.0
        self.energy_floor=0.0
        self.frame_length=25.0
        self.frame_shift=10.0
        self.high_freq=0.0
        self.htk_compat=False
        self.low_freq=20.0
        self.min_duration=0.0
        self.num_mel_bins=80
        self.output_format="kaldi"
        self.preemphasis_coefficient=0.97
        self.raw_energy=True
        self.remove_dc_offset=True
        self.round_to_power_of_two=True
        self.sample_frequency=16000.0
        self.snip_edges=True
        self.subtract_mean=False
        self.use_energy=False
        self.use_log_fbank=True
        self.use_power=True
        self.utt2spk=""
        self.vtln_high=-500.0
        self.vtln_low=100.0
        self.vtln_map=""
        self.vtln_warp=1.0
        self.window_type="povey"
        self.write_utt2dur=""

class FbankOptions():
    def __init__(self):
        self.frame_opts = FrameExtractionOptions()
        self.mel_opts = MelBanksOptions(23)
        self.use_energy = False
        self.energy_floor = 0.0
        self.raw_energy = True
        self.htk_compat = False
        self.use_log_fbank = True
        self.use_power = True
    def parse(self):        
        opts = DefaultOptions()
        return opts

    def register(self):
        opts = self.parse()
        self.frame_opts.register(opts)
        self.mel_opts.register(opts)
        self.use_energy = opts.use_energy
        self.energy_floor = opts.energy_floor
        self.raw_energy = opts.raw_energy
        self.htk_compat = opts.htk_compat
        self.use_log_fbank = opts.use_log_fbank
        self.use_power = opts.use_power

class FrameExtractionOptions():
    def __init__(self):
        self.samp_freq = 16000
        self.frame_shift_ms = 10.0
        self.frame_length_ms = 25.0
        self.dither = 1.0
        self.preemph_coeff = 0.97
        self.remove_dc_offset = True
        self.window_type = "povey"
        self.round_to_power_of_two = True
        self.blackman_coeff = 0.42
        self.snip_edges = True
        self.allow_downsample = False
        self.allow_upsample = False

    def register(self, opts):
        self.samp_freq = opts.sample_frequency
        self.frame_shift_ms = opts.frame_shift
        self.frame_length_ms = opts.frame_length
        self.dither = opts.dither
        self.preemph_coeff = opts.preemphasis_coefficient
        self.remove_dc_offset = opts.remove_dc_offset
        self.window_type = opts.window_type
        self.round_to_power_of_two = opts.round_to_power_of_two
        self.blackman_coeff = opts.blackman_coeff
        self.snip_edges = opts.snip_edges
        self.allow_downsample = opts.allow_downsample
        self.allow_upsample = opts.allow_upsample

    def window_shift(self):
        return int(self.samp_freq * 0.001 * self.frame_shift_ms)

    def window_size(self):
        return int(self.samp_freq * 0.001 * self.frame_length_ms)

    def padded_window_size(self):
        return round_up_to_nearest_power_of_two(self.window_size()) if self.round_to_power_of_two else self.window_size()

class MelBanksOptions():
    def __init__(self, num_bins=25):
        self.num_bins = num_bins
        self.low_freq = 20
        self.high_freq = 0
        self.vtln_low = 100
        self.vtln_high = -500
        self.debug_mel = False
        self.htk_mode = False

    def register(self, opts):
        self.num_bins = opts.num_mel_bins
        self.low_freq = opts.low_freq
        self.high_freq = opts.high_freq
        self.vtln_low = opts.vtln_low
        self.vtln_high = opts.vtln_high
        self.debug_mel = opts.debug_mel

class MelBanks():
    def __init__(self, opts, frame_opts, vtln_warp_factor):
        self.htk_mode = opts.htk_mode
        num_bins = opts.num_bins
        if num_bins<3:
            raise Exception("Must have at least 3 mel bins")

        sample_freq = frame_opts.samp_freq
        window_length_padded = frame_opts.padded_window_size()
        assert window_length_padded % 2 == 0
        num_fft_bins = int(window_length_padded / 2)
        nyquist = sample_freq * 0.5

        low_freq = opts.low_freq
        if opts.high_freq > 0.0:
            high_freq = opts.high_freq
        else:
            high_freq = nyquist + opts.high_freq

        if low_freq < 0.0 or low_freq >= nyquist or high_freq <= 0.0 or high_freq > nyquist or high_freq <= low_freq:
            raise Exception(f'Bad values in options: low-freq {low_freq} and high-freq {high_freq} vs. nyquist {nyquist}')

        fft_bin_width = sample_freq / window_length_padded
        mel_low_freq = self.mel_scale(low_freq)
        mel_high_freq = self.mel_scale(high_freq)

        self.debug = opts.debug_mel
        mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

        vtln_low = opts.vtln_low
        vtln_high = opts.vtln_high

        if vtln_high < 0.0:
            vtln_high += nyquist

        if vtln_warp_factor != 1.0 and (vtln_low < 0.0 or vtln_low <= low_freq or vtln_low >= high_freq or vtln_high <= 0.0 or vtln_high >= high_freq or vtln_high <= vtln_low):
            raise Exception(f"Bad values in options: vtln-low {vtln_low} and vtln-high {vtln_high}, versus low-freq {low_freq} and high-freq {high_freq}")

        self.bins = [Pair(int(), list()) for _ in range(num_bins)]
        self.center_freqs = vector(num_bins, float)

        for bin in range(num_bins):
            left_mel = mel_low_freq + bin * mel_freq_delta
            center_mel = mel_low_freq + (bin + 1) * mel_freq_delta
            right_mel = mel_low_freq + (bin + 2) * mel_freq_delta

            if vtln_warp_factor != 1.0:
                left_mel = self.vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, left_mel)
                center_mel = self.vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, center_mel)
                right_mel = self.vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, right_mel)

            self.center_freqs[bin] = self.inverse_mel_scale(center_mel)
            this_bin = np.zeros(num_fft_bins)
            first_index, last_index = -1, -1
            for i in range(num_fft_bins):
                freq = fft_bin_width * i
                mel = self.mel_scale(freq)
                if mel > left_mel and mel < right_mel:
                    weight = 0.0
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel - mel) / (right_mel - center_mel)
                    this_bin[i] = weight
                    if first_index == -1:
                        first_index = i
                    last_index = i

            assert (first_index != -1 and last_index >= first_index), "You may have set --num-mel-bins too large."

            self.bins[bin].first = first_index
            size = int(last_index + 1 - first_index)
            for i in range(first_index, first_index+size):
                self.bins[bin].second += [this_bin[i]]

            if opts.htk_mode and bin == 0 and mel_low_freq != 0.0:
                self.bins[bin].second[0] = 0.0

        if self.debug:
            for i in range(len(self.bins)):
                print(f'bin {i}, offset = {self.bins[i].first}, vec = {np.array(self.bins[i].second)}')

        return

    def inverse_mel_scale(self, mel_freq):
        return 700.0 * (np.exp(mel_freq / 1127.0) - 1.0)

    def mel_scale(self, freq):
        return 1127.0 * np.log(1.0 + freq / 700.0)

    def vtln_warp_freq(self, vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor, freq):
        if freq < low_freq or freq > high_freq:
            return freq

        assert vtln_low_cutoff > low_freq, "be sure to set the --vtln-low option higher than --low-freq"
        assert vtln_high_cutoff < high_freq, "be sure to set the --vtln-high option lower than --high-freq [or negative]"

        l = vtln_low_cutoff * max(1.0, vtln_warp_factor)
        h = vtln_high_cutoff * min(1.0, vtln_warp_factor)
        scale = 1.0 / vtln_warp_factor
        fl = scale * l
        fh = scale * h
        assert l > low_freq and h < high_freq
        scale_left = (fl - low_freq) / (l - low_freq)
        scale_right = (high_freq - fh) / (high_freq - h)

        if freq < l:
            return low_freq + scale_left * (freq - low_freq)
        elif freq < h:
            return scale * freq
        else:
            return high_freq + scale_right * (freq - high_freq)

    def vtln_warp_mel_freq(self, vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor, mel_freq):
        return self.mel_scale(self.vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor, self.inverse_mel_scale(mel_freq)))

    def compute(self, power_spectrum):
        num_bins = len(self.bins)
        mel_energies_out = np.zeros(num_bins)

        for i in range(num_bins):
            offset = self.bins[i].first
            v = self.bins[i].second
            energy = np.dot(v, power_spectrum[offset : offset + len(v)])

            if self.htk_mode and energy < 1.0:
                energy = 1.0

            mel_energies_out[i] = energy
            if self.debug:
                print("Mel Banks:")
                print(mel_energies_out)

        assert not np.isnan(mel_energies_out).any()

        return mel_energies_out

    def num_bins(self):
        return len(self.bins)

    def get_center_freqs(self):
        return self.center_freqs

    def get_bins(self):
        return self.bins

class FbankComputer():
    def __init__(self, opts=None, other=None):
        if opts != None and other == None:
            self.opts = opts
            self.srfft = None
            self.mel_banks = dict()
            if opts.energy_floor > 0.0:
                self.log_energy_floor = np.log(opts.energy_floor)
            """
            padded_window_size = opts.frame_opts.padded_window_size()
            if ((padded_window_size & (padded_window_size-1)) == 0):
                self.srfft = SplitRadixRealFFT(padded_window_size)
            """
            self.__get_mel_banks(1.0)
        elif opts == None and other != None:
            self.opts = other.opts
            self.log_energy_floor = other.log_energy_floor
            self.mel_banks = other.mel_banks
            self.srfft = None
            other.mel_banks = sorted(other.mel_banks.items())
            for key, value in self.mel_banks.items():
                self.mel_banks[key] = MelBanks(value)
            self.srrt = None

        return

    def dim(self):
        return self.opts.mel_opts.num_bins + (1 if self.opts.use_energy else 0)

    def need_raw_log_energy(self):
        return self.opts.use_energy and self.opts.raw_energy

    def get_frame_options(self):
        return self.opts.frame_opts

    def compute(self, signal_raw_log_energy, vtln_warp, signal_frame):
        MEL_BANKS = self.__get_mel_banks(vtln_warp)
        assert len(signal_frame) == self.opts.frame_opts.padded_window_size()
        feature = np.zeros(self.dim())

        if self.opts.use_energy and not self.opts.raw_energy:
            signal_raw_log_energy = np.log(max(np.dot(signal_frame, signal_frame), np.finfo(np.float32).eps))

        signal_frame = rfft(signal_frame, self.opts.frame_opts.padded_window_size())
        power_spectrum = np.power(np.abs(signal_frame), 2)

        if not self.opts.use_power:
            power_spectrum = np.sqrt(power_spectrum)

        mel_offset = 1 if (self.opts.use_energy and not self.opts.htk_compat) else 0
        mel_energies = MEL_BANKS.compute(power_spectrum)

        if self.opts.use_log_fbank:
            mel_energies = np.clip(mel_energies, a_min=np.finfo(np.float32).eps, a_max=None)
            mel_energies = np.log(mel_energies)

        if self.opts.use_energy:
            if self.opts.energy_floor > 0.0 and signal_raw_log_energy < self.log_energy_floor:
                signal_raw_log_energy = self.log_energy_floor
            energy_index = self.opts.mel_opts.num_bins if self.opts.htk_compat else 0
            feature[energy_index] = signal_raw_log_energy
        feature[mel_offset : mel_offset + self.opts.mel_opts.num_bins] = mel_energies

        return feature

    def __get_mel_banks(self, vtln_warp):
        this_mel_banks = None
        key_list = list(self.mel_banks.keys())
        try:
            vtln_warp_index = key_list.index(vtln_warp)
        except ValueError:
            vtln_warp_index = len(key_list)

        if vtln_warp_index == len(key_list):
            this_mel_banks = MelBanks(self.opts.mel_opts, self.opts.frame_opts, vtln_warp)
            self.mel_banks[vtln_warp] = this_mel_banks
        else:
            this_mel_banks = self.mel_banks[vtln_warp]
        return this_mel_banks

class Fbank():
    def __init__(self, opts):
        self.computer = FbankComputer(opts)
        self.feature_window_option = self.computer.get_frame_options()

    def dim(self):
        return self.computer.dim()

    def compute(self, wave, vtln_warp):
        rows_out = num_frames(len(wave), self.computer.get_frame_options())
        cols_out = self.computer.dim()
        if rows_out == 0:
            return None
        output = np.zeros((rows_out, cols_out))
        use_raw_log_energy = self.computer.need_raw_log_energy()
        for r in range(rows_out):
            raw_log_energy = 0.0
            window, raw_log_energy = extract_window(0, wave, r, self.computer.get_frame_options(), self.feature_window_option, (raw_log_energy if use_raw_log_energy else None))
            # FbankComputer.compute
            output[r,:] = self.computer.compute(raw_log_energy, vtln_warp, window)

        return output

    def compute_features(self, wave, sample_freq, vtln_warp):
        new_sample_freq = self.computer.get_frame_options().samp_freq
 #       print(new_sample_freq, sample_freq)
        if sample_freq == new_sample_freq:
            output = self.compute(wave, vtln_warp)
        else:
            if new_sample_freq < sample_freq and not self.computer.get_frame_options().allow_downsample:
                raise Exception(f"Waveform and config sample Frequency mismatch: {sample_freq} .vs {new_sample_freq} (use --allow-downsample=True to allow downsampling the waveform).")
            elif new_sample_freq > sample_freq and not self.computer.get_frame_options().allow_upsample:
                raise Exception(f"Waveform and config sample Frequency mismatch: {sample_freq} .vs {new_sample_freq} (use --allow-upsample=True to allow upsampling the waveform).")
            resampled_wave = resample_waveform(sample_freq, wave, new_sample_freq)
            output = self.compute(resampled_wave, vtln_warp)
        return output


'''
if __name__ == "__main__":
    args = parse()
    sr, waveform = read("./LJ001-0001.wav")
    fbank_opts = FbankOptions()
    fbank_opts.register(args)
    fbank = Fbank(fbank_opts)
    mel_spec = fbank.compute_features(waveform, sr, args.vtln_warp)


    print(mel_spec.shape)
    for i in range(len(mel_spec)):
        print(f"Frame #{i} : ", end='')
        for j in range(args.num_mel_bins):
            print(f'{mel_spec[i][j]:.4f}, ', end='')
        print()

    np.save('mel_fbank_python.npy', mel_spec)

    # Plotting both kaldi and python values frame-by-frame for debug
    mel_spec_kaldi = np.load("mel_fbank.npy")
    vmax, vmin = np.max(mel_spec_kaldi), np.min(mel_spec_kaldi)
    fig = plt.figure(figsize=(40,20))
    plt.title("Mel-Spectrogram from python")
    plt.imshow(np.rot90(mel_spec), interpolation='nearest', aspect=1.4, vmax=vmax, vmin=vmin, cmap='binary')
    fig.savefig('Mel-Spectrogram from python.png', bbox_inches='tight', format='png')
    plt.title("Mel-Spectrogram from kaldi")
    plt.imshow(np.rot90(mel_spec_kaldi), interpolation='nearest', aspect=1.4, vmax=vmax, vmin=vmin, cmap='binary')
    fig.savefig('Mel-Spectrogram from kaldi.png', bbox_inches='tight', format='png')
    plt.close(fig)
    for i in range(len(mel_spec)):
        fig = plt.figure()
        plt.plot(mel_spec[i], label='python')
        plt.plot(mel_spec_kaldi[i], label='kaldi')
        plt.legend()
        plt.show()
        fig.savefig(f"./mel_spec_plot/mel_spec_plot_#{i}.png", format='png')
        plt.close(fig)
'''