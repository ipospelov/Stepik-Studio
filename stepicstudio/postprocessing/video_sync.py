import os
import scipy.io.wavfile
import numpy as np
import math
import audio_sync
from stepicstudio.FileSystemOperations.file_system_client import FileSystemClient
from stepicstudio.operationsstatuses.operation_result import InternalOperationResult
import logging
from django.conf import settings
from stepicstudio.operationsstatuses.statuses import ExecutionStatus


class VideosSynchronizer(object):
    def __init__(self):
        self.__fs_client = FileSystemClient()
        self.__logger = logging.getLogger('stepicstudio.postprocessing.video_sync')
        self.__fft_bin_size = 1024
        self.__overlap = 0
        self.__box_height = 512
        self.__box_width = 43
        self.__samples_per_box = 7

    def synchronize(self, video_path_1: str, video_path_2: str):
        if not self.__fs_client.validate_file(video_path_1) or \
                not self.__fs_client.validate_file(video_path_2):
            self.__logger.error('Can\'t sync non existing files: %s, %s', video_path_1, video_path_2)
            return InternalOperationResult(ExecutionStatus.FATAL_ERROR)

        wavfile_1 = self.extract_audio(video_path_1)
        # print('Exctrsacted 1')
        # raw_audio_1, rate = self.read_audio(wavfile_1)
        # print('Raw conputed 1 ')
        # ft_dict_1 = self.internal_handle(raw_audio_1)
        # print('dict computed')

        wavfile_2 = self.extract_audio(video_path_2)

        latencies, dropouts = audio_sync.AnalyzeAudios(wavfile_1, wavfile_2)
        # print('Exctrsacted 1')
        # raw_audio_2, rate = self.read_audio(wavfile_2)
        # print('Raw conputed 1 ')
        # ft_dict_2 = self.internal_handle(raw_audio_2)
        # print('dict computed')
        #
        # delay = self.determine_delay(ft_dict_1, ft_dict_2, rate)
        # self.__logger.info('Computed delay: %s', delay)
        return latencies, dropouts

    def determine_delay(self, ft_dict_1, ft_dict_2, rate):
        pairs = self.find_freq_pairs(ft_dict_1, ft_dict_2)
        delay = self.find_delay(pairs)
        samples_per_sec = float(rate) / float(self.__fft_bin_size)
        seconds = round(float(delay) / float(samples_per_sec), 4)

        if seconds > 0:
            return seconds, 0
        else:
            return 0, abs(seconds)

    def internal_handle(self, raw_audio):
        bins_dict = self.make_horiz_bins(raw_audio[:44100 * 120],
                                         self.__fft_bin_size,
                                         self.__overlap,
                                         self.__box_height)
        boxes = self.make_vert_bins(bins_dict, self.__box_width)  # box width
        ft_dict = self.find_bin_max(boxes, self.__samples_per_box)
        return ft_dict

    def extract_audio(self, video_path) -> str:
        wo_extension = os.path.splitext(video_path)[0]
        audio_output = wo_extension + 'WAV.wav'
        # extract_result = self.__fs_client.execute_command_sync(
        #     [settings.FFMPEG_PATH, '-y', '-i', video_path, '-vn', '-ac', '1', '-f', 'wav', audio_output])

        # if extract_result.status is not ExecutionStatus.SUCCESS:
        #     raise Exception(extract_result.message)

        return audio_output

    # Read file
    # INPUT: Audio file
    # OUTPUT: Sets sample rate of wav file, Returns data read from wav file (numpy array of integers)
    def read_audio(self, audio_file: str):
        # Return the sample rate (in samples/sec) and data from a WAV file
        rate, data = scipy.io.wavfile.read(audio_file)
        return data, rate

    def make_horiz_bins(self, data, fft_bin_size, overlap, box_height):
        horiz_bins = {}
        # process first sample and set matrix height
        sample_data = data[0:fft_bin_size]  # get data for first sample
        if len(sample_data) == fft_bin_size:  # if there are enough audio points left to create a full fft bin
            intensities = self.fourier(sample_data)  # intensities is list of fft results
            for i in range(len(intensities)):
                box_y = i / box_height
                if box_y in horiz_bins:
                    horiz_bins[box_y].append((intensities[i], 0, i))  # (intensity, x, y)
                else:
                    horiz_bins[box_y] = [(intensities[i], 0, i)]
        # process remainder of samples
        x_coord_counter = 1  # starting at second sample, with x index 1

        for j in range(int(fft_bin_size - overlap), len(data), int(fft_bin_size - overlap)):
            sample_data = data[j:j + fft_bin_size]
            if len(sample_data) == fft_bin_size:
                intensities = self.fourier(sample_data)
                for k in range(len(intensities)):
                    box_y = k / box_height
                    if box_y in horiz_bins:
                        horiz_bins[box_y].append((intensities[k], x_coord_counter, k))  # (intensity, x, y)
                    else:
                        horiz_bins[box_y] = [(intensities[k], x_coord_counter, k)]
            x_coord_counter += 1

        return horiz_bins

    # Compute the one-dimensional discrete Fourier Transform
    # INPUT: list with length of number of samples per second
    # OUTPUT: list of real values len of num samples per second
    def fourier(self, sample):
        mag = []
        fft_data = np.fft.fft(sample)  # Returns real and complex value pairs
        for i in range(int(len(fft_data) / 2)):
            r = fft_data[i].real ** 2
            j = fft_data[i].imag ** 2
            mag.append(round(math.sqrt(r + j), 2))

        return mag

    def make_vert_bins(self, horiz_bins, box_width):
        boxes = {}
        for key in horiz_bins.keys():
            for i in range(len(horiz_bins[key])):
                box_x = horiz_bins[key][i][1] / box_width
                if (box_x, key) in boxes:
                    boxes[(box_x, key)].append((horiz_bins[key][i]))
                else:
                    boxes[(box_x, key)] = [(horiz_bins[key][i])]

        return boxes

    def find_bin_max(self, boxes, maxes_per_box):
        freqs_dict = {}
        for key in boxes.keys():
            max_intensities = [(1, 2, 3)]
            for i in range(len(boxes[key])):
                if boxes[key][i][0] > min(max_intensities)[0]:
                    if len(max_intensities) < maxes_per_box:  # add if < number of points per box
                        max_intensities.append(boxes[key][i])
                    else:  # else add new number and remove min
                        max_intensities.append(boxes[key][i])
                        max_intensities.remove(min(max_intensities))
            for j in range(len(max_intensities)):
                if max_intensities[j][2] in freqs_dict:
                    freqs_dict[max_intensities[j][2]].append(max_intensities[j][1])
                else:
                    freqs_dict[max_intensities[j][2]] = [max_intensities[j][1]]

        return freqs_dict

    def find_freq_pairs(self, freqs_dict_orig, freqs_dict_sample):
        time_pairs = []
        for key in freqs_dict_sample.keys():  # iterate through freqs in sample
            if key in freqs_dict_orig:  # if same sample occurs in base
                for i in range(len(freqs_dict_sample[key])):  # determine time offset
                    for j in range(len(freqs_dict_orig[key])):
                        time_pairs.append((freqs_dict_sample[key][i], freqs_dict_orig[key][j]))

        return time_pairs

    def find_delay(self, time_pairs):
        t_diffs = {}
        for i in range(len(time_pairs)):
            delta_t = time_pairs[i][0] - time_pairs[i][1]
            if delta_t in t_diffs:
                t_diffs[delta_t] += 1
            else:
                t_diffs[delta_t] = 1
        t_diffs_sorted = sorted(t_diffs.items(), key=lambda x: x[1])
        time_delay = t_diffs_sorted[-1][0]

        return time_delay
