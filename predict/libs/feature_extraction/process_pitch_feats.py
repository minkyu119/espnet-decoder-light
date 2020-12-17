import numpy as np
from copy import deepcopy
import argparse
import math

def vector(size, value_type):
    return [value_type() for i in range(size)]


def matrix(row_size, col_size, value_type):
    return [[value_type() for j in range(col_size)] for i in range(row_size)]

class DefaultProcessPitch():
    def __init__(self):
        self.add_delta_pitch=True
        self.add_normalized_log_pitch=True
        self.add_pov_feature=True
        self.add_raw_log_pitch=False
        self.delay=0
        self.delta_window=2
        self.normalization_left_context=75
        self.normalization_right_context=75
        self.srand=0
        self.delta_pitch_noise_stddev=0.005
        self.delta_pitch_scale=10
        self.pitch_scale=2
        self.pov_offset=0
        self.pov_scale=2

def process_pitch(input_speech): 
    opts = DefaultProcessPitch()
    online_process_pitch = OnlineProcessPitch(opts, input_speech)
    output = matrix(online_process_pitch.num_frames_ready(), online_process_pitch.dim(), float)
    for t in range(online_process_pitch.num_frames_ready()):
        row = output[t]
        online_process_pitch.get_frame(t, row)

    return np.array(output)

def nccf_to_pov_feature(n):
    n = np.clip(n, a_min=-1.0, a_max=1.0)
    f = np.power((1.0001 - n), 0.15) - 1.0
    assert f-f == 0
    return f

def nccf_to_pov(n):
    n = np.clip(n, a_min=-1.0, a_max=1.0)
    ndash = np.abs(n)

    r = -5.2 + 5.4 * np.exp(7.5 * (ndash - 1.0)) + 4.8 * ndash - 2.0 * np.exp(-10.0 * ndash) + 4.2 * np.exp(20.0 * (ndash - 1.0))
    p = 1.0 / (1 + np.exp(-1.0 * r))
    assert p - p == 0
    return p


class DeltaFeaturesOptions():
    def __init__(self, order=2, window=2):
        self.order = order
        self.window = window
        return

class DeltaFeatures():
    def __init__(self, opts):
        self.opts = opts
        assert opts.order >= 0 and opts.order < 1000
        assert opts.window > 0 and opts.window < 1000

        self.scales = vector(opts.order+1, list)
        self.scales[0].append(1.0)

        for i in range(1, opts.order+1):
            prev_scales = self.scales[i-1]
            cur_scales = self.scales[i]
            window = opts.window
            assert window != 0
            prev_offset = int(int(len(prev_scales)-1)/2)
            cur_offset = prev_offset + window
            cur_scales[:] = vector(len(prev_scales) + 2 * window, float)[:]

            normalizer = 0.0
            for j in range(-window, window+1):
                normalizer += j*j
                for k in range(-prev_offset, prev_offset+1):
                    cur_scales[j+k+cur_offset] += prev_scales[k+prev_offset] * j

            for e in range(len(cur_scales)):
                cur_scales[e] *= (1.0 / normalizer)

        return

    def process(self, input_feats, frame, output_frame):
        assert frame < len(input_feats)
        num_frames = len(input_feats)
        feat_dim = len(input_feats[0])
        assert len(output_frame) == feat_dim * (self.opts.order+1)

        for i in range(self.opts.order+1):
            scales = self.scales[i]
            max_offset = int((len(scales)-1)/2)
            for j in range(-max_offset, max_offset+1):
                offset_frame = frame + j
                if offset_frame < 0 :
                    offset_frame = 0
                elif offset_frame >= num_frames :
                    offset_frame = num_frames - 1
                scale = scales[j+max_offset]
                if scale != 0.0:
                    for k in range(feat_dim):
                         output_frame[k+(i*feat_dim)] += scale * input_feats[offset_frame][k]
        return


def compute_deltas(delta_opts, input_features):
    num_input_rows = len(input_features)
    num_input_cols = len(input_features[0])
    output_features = matrix(num_input_rows, num_input_cols * (delta_opts.order + 1), float)
    delta = DeltaFeatures(delta_opts)
    for r in range(num_input_rows):
        row = output_features[r]
        delta.process(input_features, r, row)


    return output_features


class NormalizationStats():
    def __init__(self):
        self.cur_num_frames = -1
        self.input_finished = False
        self.sum_pov = 0.0
        self.sum_log_pitch_pov = 0.0

    def copy_from_other(self, other):
        assert isinstance(other, NormalizationStats)
        self.cur_num_frames = other.cur_num_frames
        self.input_finished = other.input_finished
        self.sum_pov = other.sum_pov
        self.sum_log_pitch_pov = other.sum_log_pitch_pov


class OnlineProcessPitch():
    def __init__(self, opts, src):
        self.opts = opts
        self.src = src
        self.dim_ = 1 if opts.add_pov_feature else 0
        self.dim_ += 1 if opts.add_normalized_log_pitch else 0
        self.dim_ += 1 if opts.add_delta_pitch else 0
        self.dim_ += 1 if opts.add_raw_log_pitch else 0

        assert self.dim_ > 0
        assert len(self.src[0]) == 2

        self.delta_feature_noise = list()
        self.normalization_stats = list()

        return

    def dim(self):
        return self.dim_

    def is_last_frame(self, frame):
        if frame <= -1:
            return self.src.shape[0] == 0
        elif frame < self.opts.delay:
            return False if self.src.shape[0] == 0 else self.src.shape[0] == 1
        else:
            return self.src.shape[0] == (frame - self.opts.delay + 1)

    def frame_shift_in_seconds(self):
        return 0.01

    def num_frames_ready(self):
        src_frames_ready = len(self.src)
        if src_frames_ready == 0:
            return 0
        elif len(self.src) == src_frames_ready:
            return src_frames_ready + self.opts.delay
        else:
            return max(0, src_frames_ready - self.opts.normalization_right_context + self.opts.delay)

    def get_frame(self, frame, feat):
        frame_delayed = 0 if frame < self.opts.delay else frame - self.opts.delay
        assert len(feat) == self.dim_ and frame_delayed < self.num_frames_ready()

        index = 0
        if self.opts.add_pov_feature:
            feat[index] = self.__get_pov_feature(frame_delayed)
            index += 1
        if self.opts.add_normalized_log_pitch:
            feat[index] = self.__get_normalized_log_pitch_feature(frame_delayed)
            index += 1
        if self.opts.add_delta_pitch:
            feat[index] = self.__get_delta_pitch_feature(frame_delayed)
            index += 1
        if self.opts.add_raw_log_pitch:
            feat[index] = self.__get_raw_log_pitch_feature(frame_delayed)
        assert index == self.dim_
        return

    def __get_pov_feature(self, frame):
        nccf = self.src[frame][0]
        return self.opts.pov_scale * nccf_to_pov_feature(nccf) + self.opts.pov_offset

    def __get_delta_pitch_feature(self, frame):
        context = self.opts.delta_window
        start_frame = max(0, frame-context)
        end_frame = min(frame+context+1, len(self.src))
        frames_in_window = end_frame - start_frame
        feats = matrix(frames_in_window, 1, float)

        for f in range(start_frame, end_frame):
            feats[f-start_frame][0] = self.__get_raw_log_pitch_feature(f)
        delta_opts = DeltaFeaturesOptions(order=1, window=self.opts.delta_window)
        delta_feats = compute_deltas(delta_opts, feats)
        while len(self.delta_feature_noise) <= frame:
            rand_uniform = (np.random.randint(32767) + 1.0) / (32767+2.0)
            rand_gauss = np.sqrt(-2 * np.log(rand_uniform)) * np.cos(2*np.pi*rand_uniform)
            self.delta_feature_noise.append(rand_gauss * self.opts.delta_pitch_noise_stddev)

        return (delta_feats[frame-start_frame][1] + self.delta_feature_noise[frame]) * self.opts.delta_pitch_scale
        #return delta_feats[frame-start_frame][1] * self.opts.delta_pitch_scale

    def __get_raw_log_pitch_feature(self, frame):
        pitch = self.src[frame][1]
        assert pitch > 0
        return np.log(pitch)

    def __get_normalized_log_pitch_feature(self, frame):
        self.__update_normalization_stats(frame)
        log_pitch = self.__get_raw_log_pitch_feature(frame)
        avg_log_pitch = self.normalization_stats[frame].sum_log_pitch_pov / self.normalization_stats[frame].sum_pov
        normalized_log_pitch = log_pitch - avg_log_pitch
        return normalized_log_pitch * self.opts.pitch_scale

    def __get_normalization_window(self, t, src_frames_ready):
        left_context = self.opts.normalization_left_context
        right_context = self.opts.normalization_right_context
        window_begin = max(0, t-left_context)
        window_end = min(t+right_context+1, src_frames_ready)
        return window_begin, window_end

    def __update_normalization_stats(self, frame):
        assert frame >= 0
        if len(self.normalization_stats) <= frame:
            for i in range(len(self.normalization_stats), frame+2):
                self.normalization_stats.append(NormalizationStats())
        cur_num_frames = len(self.src)
        input_finished = cur_num_frames == len(self.src)

        this_stats = self.normalization_stats[frame]
        if this_stats.cur_num_frames == cur_num_frames and this_stats.input_finished == input_finished:
            return

        this_window_begin, this_window_end = self.__get_normalization_window(frame, cur_num_frames)
        if frame>0 :
            prev_stats = self.normalization_stats[frame-1]
            if prev_stats.cur_num_frames == cur_num_frames and prev_stats.input_finished == input_finished:
                this_stats.copy_from_other(prev_stats)
                prev_window_begin, prev_window_end = self.__get_normalization_window(frame-1, cur_num_frames)
                if this_window_begin != prev_window_begin:
                    assert this_window_begin == prev_window_begin + 1
                    nccf, pitch = self.src[prev_window_begin]
                    accurate_pov = nccf_to_pov(nccf)
                    log_pitch = np.log(pitch)
                    this_stats.sum_pov -= accurate_pov
                    this_stats.sum_log_pitch_pov -= accurate_pov * log_pitch
                if this_window_end != prev_window_end:
                    assert this_window_end == prev_window_end + 1
                    nccf, pitch = self.src[prev_window_end]
                    accurate_pov = nccf_to_pov(nccf)
                    log_pitch = np.log(pitch)
                    this_stats.sum_pov += accurate_pov
                    this_stats.sum_log_pitch_pov += accurate_pov * log_pitch
                return

        this_stats.cur_num_frames = cur_num_frames
        this_stats.input_finished = input_finished
        this_stats.sum_pov = 0.0
        this_stats.sum_log_pitch_pov = 0.0
        for f in range(this_window_begin, this_window_end):
            nccf, pitch = self.src[f]
            accurate_pov = nccf_to_pov(nccf)
            log_pitch = np.log(pitch)
            this_stats.sum_pov += accurate_pov
            this_stats.sum_log_pitch_pov += accurate_pov * log_pitch

        return

def parse():
    parser = argparse.ArgumentParser(description='Post-process Kaldi pitch features, consisting of pitch and NCCF, into \
                                                    features suitable for input to ASR system.  Default setup produces \
                                                    3-dimensional features consisting of (pov-feature, pitch-feature, \
                                                    delta-pitch-feature), where pov-feature is warped NCCF, pitch-feature \
                                                    is log-pitch with POV-weighted mean subtraction over 1.5 second window, \
                                                    and delta-pitch-feature is delta feature computed on raw log pitch. \
                                                    In general, you can select from four features: (pov-feature, \
                                                    pitch-feature, delta-pitch-feature, raw-log-pitch), produced in that \
                                                    order, by setting the boolean options (--add-pov-feature, \
                                                    --add-normalized-log-pitch, --add-delta-pitch and --add-raw-log-pitch)')
    parser.add_argument("--add-delta-pitch", default=True, type=bool, help='If true, time derivative of log-pitch is added to output features (bool, default = true)')
    parser.add_argument("--add-normalized-log-pitch", default=True, type=bool, help='If true, the log-pitch with POV-weighted mean subtraction over 1.5 second window \
                                                                                     is added to output features (bool, default = true)')
    parser.add_argument("--add-pov-feature", default=True, type=bool, help='If true, the warped NCCF is added to output features (bool, default = true)')
    parser.add_argument("--add-raw-log-pitch", default=False, type=bool, help='If true, log(pitch) is added to output features (bool, default = false)')
    parser.add_argument("--delay", default=0, type=int, help='Number of frames by which the pitch information is delayed. (int, default = 0)')
    parser.add_argument("--delta-window", default=2, type=int, help='Number of frames on each side of central frame, to use for delta window. (int, default = 2)')
    parser.add_argument("--normalization-left-context", default=75, type=int, help='Left-context (in frames) for moving window normalization (int, default = 75)')
    parser.add_argument("--normalization-right-context", default=75, type=int, help='Right-context (in frames) for moving window normalization (int, default = 75)')
    parser.add_argument("--srand", default=0, type=int, help='Seed for random number generator, used to add noise to delta-log-pitch features (int, default = 0)')
    parser.add_argument("--delta-pitch-noise-stddev", default=0.005, type=float, help='Standard deviation for noise we add to the delta log-pitch (before scaling); \
                                                               should be about the same as delta-pitch option to pitch creation.  \
                                                               The purpose is to get rid of peaks in the delta-pitch caused by discretization of pitch values. \
                                                               (float, default = 0.005)')
    parser.add_argument("--delta-pitch-scale", default=10, type=float, help='Term to scale the final delta log-pitch feature (float, default = 10)')
    parser.add_argument("--pitch-scale", default=2, type=float, help='Scaling factor for the final normalized log-pitch value (float, default = 2)')
    parser.add_argument("--pov-offset", default=0, type=float, help='This can be used to add an offset to the POV feature. \
                                                           Intended for use in online decoding as a substitute for CMN. (float, default = 0)')
    parser.add_argument("--pov-scale", default=2, type=float, help='Scaling factor for final POV (probability of voicing) feature (float, default = 2)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    pitch_result = np.load("temp.npy")
    pitch_final_result = process_pitch(args, pitch_result)
    for i in range(len(pitch_final_result)):
        print(f'Frame #{i} : POV = {pitch_final_result[i][0]}, Norm-Log-Pitch = {pitch_final_result[i][1]}, Delta-Pitch = {pitch_final_result[i][2]}')
    np.save("pitch_python.npy", pitch_final_result)
