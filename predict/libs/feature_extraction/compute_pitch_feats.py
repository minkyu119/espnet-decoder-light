import numpy as np
from scipy.io.wavfile import read
from copy import deepcopy
import argparse
import time
import sys
import math

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.9f}".format(x)})

def vector(size, value_type):
    return [value_type() for i in range(size)]


def matrix(row_size, col_size, value_type):
    return [[value_type() for j in range(col_size)] for i in range(row_size)]

class DefaultPitchOptions():
    def __init__(self):
        self.delta_pitch=0.005
        self.frame_length=25
        self.frame_shift=10
        self.frames_per_chunk=0
        self.lowpass_cutoff=1000
        self.lowpass_filter_width=1
        self.max_f0=400
        self.max_frames_latency=0
        self.min_f0=50
        self.nccf_ballast=7000
        self.nccf_ballast_online=False
        self.penalty_factor=0.1
        self.preemphasis_coefficient=0
        self.recompute_frame=500
        self.resample_frequency=4000
        self.sample_frequency=16000
        self.simulate_first_pass_online=False
        self.snip_edges=True
        self.soft_min_f0=10
        self.upsample_filter_width=5


pitch_use_naive_search = False

def select_lags(opts):
    min_lag = 1.0 / opts.max_f0
    max_lag = 1.0 / opts.min_f0

    lags = list()
    lag = min_lag
    while lag <= max_lag:
        lags.append(lag)
        lag *= 1 + opts.delta_pitch

    return lags


def compute_local_cost(nccf_pitch, lags, opts):
    assert len(nccf_pitch) == len(lags)
    nccf_pitch_np = np.array(nccf_pitch)
    lags_np = np.array(lags)

    local_cost = np.ones(len(lags))
    local_cost += nccf_pitch_np * -1.0
    #local_cost += (lags_np * opts.soft_min_f0) + (nccf_pitch_np * 1.0)
    local_cost += opts.soft_min_f0 * lags_np * nccf_pitch_np

    return local_cost.tolist()


def approx_equal(a, b, relative_tolerance=0.01):
    if a == b:
        return True
    diff = np.abs(a - b)
    if np.isinf(diff) or np.isnan(diff):
        return False
    return diff <= relative_tolerance * (np.abs(a) + np.abs(b))


def compute_correlation(wave, first_lag, last_lag, nccf_window_size, inner_prod, norm_prod):
    zero_mean_wave = np.array(wave)
    wave_part = wave[0:nccf_window_size]

    zero_mean_wave += -1.0 * sum(wave_part) / nccf_window_size
    sub_vec1 = zero_mean_wave[0:nccf_window_size]
    e1 = np.dot(sub_vec1, sub_vec1)
    for lag in range(first_lag, last_lag+1):
        sub_vec2 = zero_mean_wave[lag:lag+nccf_window_size]
        e2 = np.dot(sub_vec2, sub_vec2)
        sum_e = np.dot(sub_vec1, sub_vec2)
        inner_prod[lag - first_lag] = sum_e
        norm_prod[lag - first_lag] = e1 * e2
    return


def compute_nccf(inner_prod, norm_prod, nccf_ballast, nccf_vec):
    assert len(inner_prod) == len(norm_prod) and len(inner_prod) == len(nccf_vec)
    for lag in range(len(inner_prod)):
        numerator = inner_prod[lag]
        denominator = math.pow(norm_prod[lag] + nccf_ballast, 0.5)
        if denominator != 0.0:
            nccf = numerator / denominator
        else:
            assert numerator == 0.0
            nccf = 0.0
        assert nccf < 1.01 and nccf > -1.01
        nccf_vec[lag] = nccf
    return


def compute_kaldi_pitch_first_pass(opts, wav, output):
    return


def compute_kaldi_pitch(wav):
    '''
    if opts.simulate_first_pass_online:
        compute_kaldi_pitch_first_pass(opts, wav, output)
        return
    '''
    print(wav)
    opts = DefaultPitchOptions()
    pitch_extractor = OnlinePitchFeatureImpl(opts)
    if opts.frames_per_chunk == 0:
        pitch_extractor.accept_waveform(opts.sample_frequency, wav)

    pitch_extractor.input_finished_()
    num_frames = pitch_extractor.num_frames_ready()
    if num_frames == 0:
        print('No frames output in pitch extraction')
        return

    output = matrix(num_frames, 2, float)
    for frame in range(num_frames):
        pitch_extractor.get_frame(frame, output[frame])
  
    return output


class ArbitraryResample():
    def __init__(self, num_samples_in, sample_rate_in, filter_cutoff, sample_points, num_zeros):
        self.__decl_parameters()
        self.num_samples_in = num_samples_in
        self.sampling_rate_in = sample_rate_in
        self.filter_cutoff = filter_cutoff
        self.num_zeros = num_zeros

        assert num_samples_in > 0 and sample_rate_in > 0.0 and filter_cutoff > 0.0 and filter_cutoff * 2.0 <= sample_rate_in and num_zeros > 0
        self.__set_indexes(sample_points)
        self.__set_weights(sample_points)

        return

    def __decl_parameters(self):
        self.num_samples_in = 0
        self.sampling_rate_in = 0
        self.filter_cutoff = 0.0
        self.num_zeros = 0
        self.first_index = []
        self.weights = []

    def num_samples_in(self):
        return self.num_samples_in

    def num_samples_out(self):
        return len(self.weights)

    def resample_matrix(self, input_matrix, output_matrix):
        input = np.array(input_matrix)
        output = np.array(output_matrix)
        assert input.shape[0] == output.shape[0] and input.shape[1] == self.num_samples_in and output.shape[1] == len(self.weights)
        output_col = np.zeros(output.shape[0])
        for i in range(self.num_samples_out()):
            input_part = input[:, self.first_index[i]:self.first_index[i]+len(self.weights[i])]
            weight_vec = self.weights[i]
            output_col = np.matmul(input_part, weight_vec)
            output[:, i] = output_col

        #output_matrix = output.tolist()
        for i in range(output.shape[0]):
            output_matrix[i][:] = output[i].tolist()[:]
        return

    def __set_indexes(self, sample_points):
        num_samples = len(sample_points)
        self.first_index = vector(num_samples, int)
        self.weights = vector(num_samples, list)

        filter_width = self.num_zeros / (2.0 * self.filter_cutoff)
        for i in range(num_samples):
            t = sample_points[i]
            t_min = t - filter_width
            t_max = t + filter_width
            index_min = math.ceil(self.sampling_rate_in * t_min)
            index_max = math.floor(self.sampling_rate_in * t_max)
            if index_min < 0 : index_min = 0
            if index_max >= self.num_samples_in : index_max = self.num_samples_in - 1
            self.first_index[i] = index_min
            self.weights[i]  = vector(index_max - index_min + 1, float)
        return

    def __set_weights(self, sample_points):
        num_samples_out = self.num_samples_out()
        for i in range(num_samples_out):
            for j in range(len(self.weights[i])):
                delta_t = sample_points[i] - (self.first_index[i] + j) / self.sampling_rate_in
                self.weights[i][j] = self.__filter_func(delta_t) / self.sampling_rate_in
        return

    def __filter_func(self, t):
        if np.abs(t) < self.num_zeros / 2 * self.filter_cutoff:
            window = 0.5 * (1 + np.cos(2 * np.pi * self.filter_cutoff / self.num_zeros * t))
        else:
            window = 0.0
        if t != 0.0:
            filter = np.sin(2 * np.pi * self.filter_cutoff * t) / (np.pi * t)
        else:
            filter = 2.0 * self.filter_cutoff
        return float(filter * window)


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

        return output.tolist()

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


class Pair():
    def __init__(self, first=None, second=None):
        self.first = first
        self.second = second


class NccfInfo():
    def __init__(self, avg_norm_prod, mean_square_energy):
        self.avg_norm_prod = avg_norm_prod
        self.mean_square_energy = mean_square_energy
        self.nccf_pitch_resampled = None


class StateInfo():
    def __init__(self):
        self.backpointer = 0
        self.pov_nccf = 0.0
        return


class PitchFrameInfo():
    def __init__(self, num_states=None, prev_info=None):
        self.__decl_parameters()
        if num_states is not None:
            self.state_info = vector(num_states, StateInfo)
            self.state_offset = 0
            self.cur_best_state = -1
            self.prev_info = None
            return
        if prev_info is not None:
            state_info_len = len(prev_info.state_info)
            self.state_info = vector(state_info_len, StateInfo)
            self.state_offset = 0
            self.cur_best_state = -1
            self.prev_info = prev_info
            return

    def __decl_parameters(self):
        self.state_info = None
        self.state_offset = None
        self.cur_best_state = None
        self.prev_info = None
        return


    def set_best_state(self, best_state, lag_nccf, index_dict):
        lag_nccf_idx = -1
        this_info = self
        while this_info is not None:
            prev_info = this_info.prev_info
            if best_state == this_info.cur_best_state:
                return
            if prev_info is not None:
                # lag_nccf[index].first : pitch 리스트인 self.lags에서 해당되는 pitch에 대한 index가 된다.
                #                         이 index에 의해 self.lags에 정의된 pitch 리스트에서 선택된 pitch가
                #                         해당 프레임의 최종 pitch 값이 된다.
                lag_nccf[lag_nccf_idx].first = int(best_state)
            state_info_index = int(best_state - this_info.state_offset)
            assert state_info_index < len(this_info.state_info)
            this_info.cur_best_state = best_state
            best_state = this_info.state_info[state_info_index].backpointer
            if prev_info is not None:
                # lag_nccf[index].second : 해당 프레임의 Normalized Cross Correlation Function 값
                lag_nccf[lag_nccf_idx].second = this_info.state_info[state_info_index].pov_nccf
            this_info = prev_info
            if this_info is not None:
                lag_nccf_idx -= 1
        return


    def compute_latency(self, max_latency):
        if max_latency <= 0: return 0

        latency = 0
        num_states = len(self.state_info)
        min_living_state = 0;
        max_living_state = num_states - 1
        this_info = self

        while this_info is not None and latency < max_latency:
            offset = this_info.state_offset
            assert min_living_state >= offset and max_living_state - offset < len(this_info.state_info)

            min_living_state = this_info.state_info[min_living_state - offset].backpointer
            max_living_state = this_info.state_info[max_living_state - offset].backpointer

            if min_living_state == max_living_state:
                return latency

            this_info = this_info.prev_info
            if this_info is not None:
                latency += 1

        return latency

    def update_previous_best_state(self):
        return

    def set_nccf_pow(self, nccf_pov):
        num_states = len(nccf_pov)
        assert num_states == len(self.state_info)
        for i in range(num_states):
            self.state_info[i].pov_nccf = nccf_pov[i]
        return


    def compute_backtraces(self, opts, nccf_pitch, lags, prev_forward_cost_vec, index_info, this_forward_cost_vec):
        num_states = len(nccf_pitch)
        local_cost = compute_local_cost(nccf_pitch, lags, opts)

        delta_pitch_sq = math.pow(math.log(1 + opts.delta_pitch), 2)
        inter_frame_factor = delta_pitch_sq * opts.penalty_factor

        for i in range(len(index_info), num_states):
            index_info.append(Pair(0, 0))
        bounds = index_info

        prev_forward_cost = np.array(prev_forward_cost_vec)
        this_forward_cost = np.array(this_forward_cost_vec)

        if pitch_use_naive_search:
            for i in range(num_states):
                best_cost = float('inf')
                best_j = -1
                for j in range(num_states):
                    this_cost = (j-i) * (j-i) * inter_frame_factor + prev_forward_cost[j]
                    if this_cost < best_cost:
                        best_cost = this_cost
                        best_j = j
                this_forward_cost[i] = best_cost
                self.state_info[i].backpointer = int(best_j)
        else:
            last_backpointer = 0
            for i in range(num_states):
                start_j = last_backpointer
                best_cost = (start_j - i) * (start_j - i) * inter_frame_factor + prev_forward_cost[start_j]
                best_j = start_j
                for j in range(start_j+1, num_states):
                    this_cost = (j-i) * (j-i) * inter_frame_factor + prev_forward_cost[j]
                    if this_cost < best_cost:
                        best_cost = this_cost
                        best_j = j
                    else: break
                self.state_info[i].backpointer = int(best_j)
                this_forward_cost[i] = best_cost
                bounds[i].first = best_j
                bounds[i].second = num_states - 1
                last_backpointer = best_j

            for iter in range(num_states):
                changed = False
                if iter % 2 == 0:
                    last_backpointer = num_states - 1
                    for i in range(num_states - 1, -1, -1):
                        lower_bound = bounds[i].first
                        upper_bound = min(last_backpointer, bounds[i].second)
                        if upper_bound == lower_bound:
                            last_backpointer = lower_bound
                            continue
                        best_cost = this_forward_cost[i]
                        best_j = self.state_info[i].backpointer
                        initial_best_j = best_j

                        if best_j == upper_bound:
                            last_backpointer = best_j
                            continue

                        for j in range(upper_bound, lower_bound+1, -1):
                            this_cost = (j-i) * (j-i) * inter_frame_factor + prev_forward_cost[j]
                            if this_cost < best_cost:
                                best_cost = this_cost
                                best_j = j
                            elif best_j > j:
                                break

                        bounds[i].second = best_j
                        if best_j != initial_best_j:
                            this_forward_cost[i] = best_cost
                            self.state_info[i].backpointer = int(best_j)
                            changed = True
                        last_backpointer = best_j
                else:
                    last_backpointer = 0
                    for i in range(num_states):
                        lower_bound = max(last_backpointer, bounds[i].first)
                        upper_bound = bounds[i].second
                        if upper_bound == lower_bound:
                            last_backpointer = lower_bound
                            continue
                        best_cost = this_forward_cost[i]
                        best_j = self.state_info[i].backpointer
                        initial_best_j = best_j

                        if best_j == lower_bound:
                            last_backpointer = best_j
                            continue

                        for j in range(lower_bound, upper_bound-1):
                            this_cost = (j-i) * (j-i) * inter_frame_factor + prev_forward_cost[j]
                            if this_cost < best_cost:
                                best_cost = this_cost
                                best_j = j
                            elif best_j < j:
                                break

                        bounds[i].first = best_j
                        if best_j != initial_best_j:
                            this_forward_cost[i] = best_cost
                            self.state_info[i].backpointer = int(best_j)
                            changed = True
                        last_backpointer = best_j

                if not changed:
                    break

        self.cur_best_state = -1
        this_forward_cost += local_cost
        this_forward_cost_vec[:] = this_forward_cost.tolist()[:]

        return


class OnlinePitchFeatureImpl():
    def __init__(self, opts):
        self.__decl_parameters()
        self.opts = opts
        self.forward_cost_remainder = 0.0
        self.input_finished = False
        self.signal_sum_square = 0.0
        self.signal_sum = 0.0
        self.downsampled_samples_processed = 0

        self.signal_resampler = LinearResample(opts.sample_frequency, opts.resample_frequency, opts.lowpass_cutoff, opts.lowpass_filter_width)

        outer_min_lag = 1 / opts.max_f0 - (opts.upsample_filter_width / (2 * opts.resample_frequency))
        outer_max_lag = 1 / opts.min_f0 + (opts.upsample_filter_width / (2 * opts.resample_frequency))

        self.nccf_first_lag = math.ceil(opts.resample_frequency * outer_min_lag)
        self.nccf_last_lag = math.floor(opts.resample_frequency * outer_max_lag)
        self.frames_latency = 0
        self.lags = select_lags(opts)

        upsample_cutoff = opts.resample_frequency * 0.5
        lags_offset = [self.lags[i] + (-self.nccf_first_lag / opts.resample_frequency) for i in range(len(self.lags))]

        num_measured_lags = self.nccf_last_lag + 1 - self.nccf_first_lag
        self.nccf_resampler = ArbitraryResample(num_measured_lags, opts.resample_frequency,
                                           upsample_cutoff, lags_offset,
                                           opts.upsample_filter_width)

        self.frame_info = list()
        self.frame_info.append(PitchFrameInfo(num_states=len(self.lags)))
        self.forward_cost = vector(len(self.lags), float)
        return

    def __decl_parameters(self):
        self.nccf_first_lag = 0
        self.nccf_last_lag = 0
        self.lags = []
        self.nccf_resampler = None
        self.frame_info = []
        self.nccf_info = []
        self.frames_latency = 0
        self.forward_cost = []
        self.forward_cost_remainder = 0.0
        self.lag_nccf = []
        self.input_finished = False
        self.signal_sum_square = 0.0
        self.signal_sum = 0.0
        self.downsampled_samples_processed = 0
        self.downsampled_signal_remainder = []
        return

    # public:
    def dim(self):
        return 2

    def frame_shift_in_seconds(self):
        return self.opts.frame_shift / 1000.

    def num_frames_ready(self):
        num_frames = len(self.lag_nccf)
        latency = self.frames_latency
        assert latency <= num_frames
        return num_frames - latency

    def is_last_frame(self, frame):
        t = self.num_frames_ready()
        assert frame < t
        return self.input_finished and (frame+1 == t)

    def get_frame(self, frame, feat):
        assert frame < self.num_frames_ready() and len(feat) == 2
        feat[0] = self.lag_nccf[frame].second
        feat[1] = 1 / self.lags[self.lag_nccf[frame].first]
        return

    def accept_waveform(self, sampling_rate, wave):
        FLUSH = self.input_finished
        start = time.time()
        downsampled_wave = self.signal_resampler.resample(wave, FLUSH)
        end = time.time()
        print(f"Downsampling input waveform signal : {end-start} second")

        cur_sum_square = self.signal_sum_square
        cur_sum = self.signal_sum
        cur_num_sample = self.downsampled_samples_processed
        prev_frame_end_sample = 0

        if not self.opts.nccf_ballast_online:
            cur_sum_square += np.dot(downsampled_wave, downsampled_wave)
            cur_sum += sum(downsampled_wave)
            cur_num_sample += len(downsampled_wave)

        end_frame = self.__num_frames_available(self.downsampled_samples_processed + len(downsampled_wave), self.opts.snip_edges)
        start_frame = len(self.frame_info) - 1
        num_new_frames = end_frame - start_frame

        if num_new_frames == 0:
            self.__update_remainder(downsampled_wave)
            return

        num_measured_lags = self.nccf_last_lag + 1 - self.nccf_first_lag
        num_resampled_lags = len(self.lags)
        frame_shift = int(self.opts.resample_frequency * self.opts.frame_shift / 1000.0)
        basic_frame_length = int(self.opts.resample_frequency * self.opts.frame_length / 1000.0)
        full_frame_length = basic_frame_length + self.nccf_last_lag

        window = vector(full_frame_length, float)
        inner_prod = vector(num_measured_lags, float)
        norm_prod = vector(num_measured_lags, float)

        nccf_pitch = matrix(num_new_frames, num_measured_lags, float)
        nccf_pov = matrix(num_new_frames, num_measured_lags, float)

        cur_forward_cost = vector(num_resampled_lags, float)

        start = time.time()
        windows = []
        for frame in range(start_frame, end_frame):
            start_sample = 0
            if self.opts.snip_edges:
                start_sample = frame * frame_shift
            else:
                start_sample = int((frame+0.5) * frame_shift - full_frame_length / 2)
            window = self.__extract_frame(downsampled_wave, start_sample, window)
            windows.append(window)
            if self.opts.nccf_ballast_online:
                end_sample = start_sample + full_frame_length - self.downsampled_samples_processed
                assert end_sample > 0

                if end_sample > len(downsampled_wave):
                    assert self.input_finished
                    end_sample = len(downsampled_wave)

                new_part_start = prev_frame_end_sample
                new_part_end = new_part_start + (end_sample - prev_frame_end_sample)
                new_part = downsampled_wave[new_part_start:new_part_end]

                cur_num_sample += len(new_part)
                cur_sum_square += np.dot(new_part, new_part)
                cur_sum += sum(new_part)
                prev_frame_end_sample = end_sample

            mean_square = cur_sum_square / cur_num_sample - math.pow(cur_sum / cur_num_sample, 2)
            compute_correlation(window, self.nccf_first_lag, self.nccf_last_lag, basic_frame_length, inner_prod, norm_prod)
            nccf_ballast_pov = 0.0
            nccf_ballast_pitch = math.pow(mean_square * basic_frame_length, 2) * self.opts.nccf_ballast
            avg_norm_prod = sum(norm_prod) / len(norm_prod)

            nccf_pitch_row = nccf_pitch[frame - start_frame]
            compute_nccf(inner_prod, norm_prod, nccf_ballast_pitch, nccf_pitch_row)
            nccf_pov_row = nccf_pov[frame - start_frame]
            compute_nccf(inner_prod, norm_prod, nccf_ballast_pov, nccf_pov_row)

            if frame < self.opts.recompute_frame:
                self.nccf_info.append(NccfInfo(avg_norm_prod, mean_square))
        end = time.time()
        print(f"Compute NCCF from windowed downsampled signal : {end-start} seconds")

        nccf_pitch_np = np.array(nccf_pitch)
        nccf_pov_np = np.array(nccf_pov)

        start = time.time()
        nccf_pitch_resampled = matrix(num_new_frames, num_resampled_lags, float)
        self.nccf_resampler.resample_matrix(nccf_pitch, nccf_pitch_resampled)
        del nccf_pitch
        nccf_pov_resampled = matrix(num_new_frames, num_resampled_lags, float)
        self.nccf_resampler.resample_matrix(nccf_pov, nccf_pov_resampled)
        del nccf_pov
        end = time.time()
        print(f"Down-Sampling NCCF : {end-start} seconds")

        self.__update_remainder(downsampled_wave)

        start = time.time()
        index_info = []
        forward_cost_total = []
        for frame in range(start_frame, end_frame):
            frame_idx = frame - start_frame
            prev_info = self.frame_info[-1]
            cur_info = PitchFrameInfo(prev_info=prev_info)
            cur_info.set_nccf_pow(nccf_pov_resampled[frame_idx])
            cur_info.compute_backtraces(self.opts, nccf_pitch_resampled[frame_idx],
                                        self.lags, self.forward_cost, index_info,
                                        cur_forward_cost)

            # TODO : Make sure the value swap implemented properly
            temp = np.array(cur_forward_cost)
            cur_forward_cost[:] = self.forward_cost[:]
            self.forward_cost[:] = temp.tolist()[:]

            remainder = min(self.forward_cost)
            self.forward_cost_remainder += remainder
            for i in range(len(self.forward_cost)):
                self.forward_cost[i] += (-remainder)

            self.frame_info.append(cur_info)
            if frame < self.opts.recompute_frame:
                self.nccf_info[frame].nccf_pitch_resampled = deepcopy(nccf_pitch_resampled[frame_idx])
            if frame == self.opts.recompute_frame - 1 and not self.opts.nccf_ballast_online:
                self.__recompute_backtraces()
        end = time.time()
        print(f"Compute cost with Viterbi Algorithm for selecting proper pitch : {end-start} second")

        best_final_state = self.forward_cost.index(min(self.forward_cost))
        for i in range(len(self.lag_nccf), len(self.frame_info)-1):
            self.lag_nccf.append(Pair(0, 0.0))

        frame_info_index = dict()
        for i, info in enumerate(self.frame_info):
            info_id = id(info)
            frame_info_index[info_id] = i

        self.frame_info[-1].set_best_state(best_final_state, self.lag_nccf, frame_info_index)
        self.frames_latency = self.frame_info[-1].compute_latency(self.opts.max_frames_latency)
        return

    def input_finished_(self):
        self.input_finished = True
        self.accept_waveform(self.opts.sample_frequency, np.array([]))
        num_frames = len(self.frame_info) - 1
        if num_frames < self.opts.recompute_frame and not self.opts.nccf_ballast_online:
            self.__recompute_backtraces()
        self.frames_latency = 0
        print(f'Pitch-Tracking Viterbi cost is {self.forward_cost_remainder / num_frames} per frame, over {num_frames} frames.')
        return

    # private:
    def __num_frames_available(self, num_downsampled_samples, snip_edges):
        frame_shift = int(self.opts.resample_frequency * self.opts.frame_shift / 1000.0)
        frame_length = int(self.opts.resample_frequency * self.opts.frame_length / 1000.0)

        if not self.input_finished:
            frame_length += self.nccf_last_lag

        if num_downsampled_samples < frame_length:
            return 0
        else:
            if not snip_edges:
                if self.input_finished:
                    return int(num_downsampled_samples * 1 / frame_shift_in_seconds + 0.5)
                else:
                    return int((num_downsampled_samples - frame_length / 2) * 1 / frame_shift + 0.5)
            else:
                return int((num_downsampled_samples - frame_length) / frame_shift + 1)


    def __extract_frame(self, downsampled_wave_part, sample_index, window):
        full_frame_length = len(window)
        offset = sample_index - self.downsampled_samples_processed

        if sample_index < 0:
            assert self.opts.snip_edges == False
            sub_frame_length = sample_index + full_frame_length
            sub_frame_index = full_frame_length - sub_frame_length
            assert sub_frame_length > 0 and sub_frame_index > 0
            window = vector(len(window), float)
            sub_window_start = sub_frame_index
            sub_window_end = sub_frame_index+sub_frame_length
            window[sub_window_start:sub_window_end] = self.__extract_frame(downsampled_wave_part, 0, window[sub_window_start:sub_window_end])
            return window[:]

        if offset+full_frame_length > len(downsampled_wave_part):
            assert self.input_finished
            sub_frame_length = len(downsampled_wave_part) - offset
            assert sub_frame_length > 0
            window = vector(len(window), float)
            sub_window_start = 0
            sub_window_end = sub_frame_length
            window[sub_window_start:sub_window_end] = self.__extract_frame(downsampled_wave_part, sample_index, window[sub_window_start:sub_window_end])
            return window[:]

        if offset >= 0:
            window[:] = downsampled_wave_part[offset:offset+full_frame_length]
        else:
            remainder_offset = len(self.downsampled_signal_remainder) + offset
            assert remainder_offset >= 0
            assert offset + full_frame_length > 0

            old_length = -offset
            new_length = offset + full_frame_length
            window[0:old_length] = self.downsampled_signal_remainder[remainder_offset : remainder_offset+old_length]
            window[old_length:old_length+new_length] = downsampled_wave_part[0:new_length]

        if self.opts.preemphasis_coefficient != 0:
            preemph_coeff = self.opts.preemphasis_coefficient
            for i in range(len(window)-1, 0, -1):
                window[i] -= preemph_coeff * window[i-1]
            window[0] *= 1 - preemph_coeff

        return window[:]


    def __recompute_backtraces(self):
        assert not self.opts.nccf_ballast_online
        num_frames = len(self.frame_info) - 1

        assert num_frames <= self.opts.recompute_frame
        assert len(self.nccf_info) == num_frames

        if num_frames == 0:
            return
        num_sample = self.downsampled_samples_processed
        sum = self.signal_sum
        sum_square = self.signal_sum_square
        mean = sum / num_sample
        mean_square = sum_square / num_sample - mean * mean

        must_recompute = False
        threshold = 0.01
        for frame in range(num_frames):
            if not approx_equal(self.nccf_info[frame].mean_square_energy, mean_square, threshold):
                must_recompute = True

        if not must_recompute:
            print("must_recompute is false")
            self.nccf_info = []
            return

        num_states = len(self.forward_cost)
        basic_frame_length = int(self.opts.resample_frequency * self.opts.frame_length / 1000.0)
        new_nccf_ballast = math.pow(mean_square*basic_frame_length, 2) * self.opts.nccf_ballast

        forward_cost_remainder = 0.0
        forward_cost = vector(num_states, float)
        next_forward_cost = forward_cost[:]
        index_info = []

        for frame in range(num_frames):
            nccf_info_frame = self.nccf_info[frame]
            old_mean_square = self.nccf_info[frame].mean_square_energy
            avg_norm_prod = self.nccf_info[frame].avg_norm_prod
            old_nccf_ballast = math.pow(old_mean_square * basic_frame_length, 2) * self.opts.nccf_ballast
            nccf_scale = math.pow((old_nccf_ballast + avg_norm_prod) / (new_nccf_ballast + avg_norm_prod), 0.5)

            for i in range(len(nccf_info_frame.nccf_pitch_resampled)):
                nccf_info_frame.nccf_pitch_resampled[i] *= nccf_scale

            self.frame_info[frame + 1].compute_backtraces(self.opts, nccf_info_frame.nccf_pitch_resampled, self.lags,
                                                          forward_cost, index_info, next_forward_cost)
            forward_cost, next_forward_cost = next_forward_cost, forward_cost
            remainder = min(forward_cost)
            forward_cost_remainder += remainder
            for i in range(len(forward_cost)):
                forward_cost[i] -= remainder

        print(f'Forward-cost per frame changed from {self.forward_cost_remainder / num_frames} to {forward_cost_remainder / num_frames}')

        self.forward_cost_remainder = forward_cost_remainder
        self.forward_cost, forward_cost = forward_cost, self.forward_cost

        if len(self.lag_nccf) != num_frames:
            num_append = (num_frames) - len(self.lag_nccf)
            for i in range(num_append):
                self.lag_nccf.append(Pair(0, 0.0))

        self.frame_info[-1].set_best_state(None, self.lag_nccf)
        self.frames_latency = self.frame_info[-1].compute_latency(self.opts.max_frames_latency)
        self.nccf_info = []

        return


    def __update_remainder(self, downsampled_wave_part):
        num_frames = len(self.frame_info) - 1
        next_frame = num_frames
        frame_shift = int(self.opts.resample_frequency * self.opts.frame_shift / 1000.0)
        next_frame_sample = frame_shift * next_frame

        self.signal_sum_square += float(np.dot(downsampled_wave_part, downsampled_wave_part))
        self.signal_sum += float(sum(downsampled_wave_part))

        next_downsampled_samples_processed = self.downsampled_samples_processed + len(downsampled_wave_part)

        if next_frame_sample > next_downsampled_samples_processed:
            full_frame_length = int(self.opts.resample_frequency * self.opts.frame_length / 1000.0) + self.nccf_last_lag
            assert full_frame_length < frame_shift
            self.downsampled_signal_remainder = []
        else:
            new_remainder = vector(next_downsampled_samples_processed - next_frame_sample, float)
            for i in range(next_frame_sample, next_downsampled_samples_processed):
                if i >= self.downsampled_samples_processed:
                    new_remainder[i - next_frame_sample] = downsampled_wave_part[i - self.downsampled_samples_processed]
                else:
                    new_remainder[i - next_frame_sample] = self.downsampled_signal_remainder[i - self.downsampled_samples_processed + len(self.downsampled_signal_remainder)]
            self.downsampled_signal_remainder, new_remainder = new_remainder, self.downsampled_signal_remainder

        self.downsampled_samples_processed = next_downsampled_samples_processed
        return

if __name__ == "__main__":
    args = parse()
    #wav, sr = sf.read('/home/johnlim/LJ001-0001.wav')
    sr, wav = read('./LJ001-0001.wav')
    pitch_output = compute_kaldi_pitch(args, wav)
    pitch_output = np.array(pitch_output)
    print(pitch_output.shape)
    np.save('temp.npy', pitch_output)
