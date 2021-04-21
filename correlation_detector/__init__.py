import datetime
import logging
import re
from collections import OrderedDict
from math import log10
from pathlib import Path
from typing import Tuple, Dict, List, Generator, Iterable, Callable

import bottleneck as bn
import numpy as np
import pandas as pd
from obspy import read, Stream, Trace
from scipy.signal import find_peaks

TemplateReadTuple = Tuple[int, Stream, Dict[str, int], float]
CorrelationFix = Tuple[str, float, int]
Event = Tuple[int, datetime.datetime, float, float, float, float, List[CorrelationFix]]


def read_data(path: Path, freqmin: float = 3.0, freqmax: float = 8.0) -> Stream:
    logging.info(f"Reading continuous data from {path}")
    with path.open('rb') as file:
        data = read(file, dtype=np.float32)
    data.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)
    starttime = min(trace.stats.starttime for trace in data)
    endtime = max(trace.stats.endtime for trace in data)
    data.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
    return data


def read_templates(templates_directory: Path, ttimes_directory: Path,
                   catalog_path: Path) -> Generator[TemplateReadTuple, None, None]:
    logging.info(f"Reading catalog from {catalog_path}")
    template_magnitudes = pd.read_csv(catalog_path, sep=r'\s+', usecols=(5,), squeeze=True, dtype=float)
    logging.info(f"Reading travel times from {ttimes_directory}")
    logging.info(f"Reading templates from {templates_directory}")
    file_regex = re.compile(r'(?P<template_number>\d+).ttimes')
    for ttimes_path in ttimes_directory.glob('*.ttimes'):
        match = file_regex.match(ttimes_path.name)
        if match:
            template_number = int(match.group('template_number'))
            try:
                logging.debug(f"Reading {ttimes_path}")
                travel_times = OrderedDict()
                with open(ttimes_path, "r") as ttimes_file:
                    while line := ttimes_file.readline():
                        key, value_string = line.split(' ')
                        network, station, channel = key.split('.')
                        trace_id = f"{network}.{station}..{channel}"
                        value = float(value_string)
                        travel_times[trace_id] = value
                template_path = templates_directory / f"{template_number}.mseed"
                logging.debug(f"Reading {template_path}")
                with template_path.open('rb') as template_file:
                    template_stream = read(template_file, dtype=np.float32)
                yield template_number, template_stream, travel_times, template_magnitudes.iloc[template_number - 1]
            except OSError as err:
                logging.warning(f"{err} occurred while reading template {template_number}")
                continue


def correlation_detector(whole_data: Stream, whole_template: Stream, all_travel_times: Dict[str, float],
                         template_magnitude: float, max_channels: int = 16, threshold_factor: float = 8.0,
                         distance_factor: float = 2.0, tolerance: int = 6, magnitude_threshold: float = 2.0,
                         cc_min_std_factor: float = 0.25, cc_max_std_factor: float = 1.5,
                         mapf: Callable = map) -> Generator[Event, None, None]:
    correlations, data, template, travel_times = preprocess(whole_data, whole_template, all_travel_times,
                                                            max_channels=max_channels,
                                                            cc_min_std_factor=cc_min_std_factor,
                                                            cc_max_std_factor=cc_max_std_factor, mapf=mapf)
    mean_correlation = bn.nanmean([trace.data for trace in correlations], axis=0)
    correlation_dmad = bn.nanmean(np.abs(mean_correlation - bn.nanmedian(mean_correlation)))
    threshold = threshold_factor * correlation_dmad
    distance = int(distance_factor * sum(trace.stats.npts for trace in template) / len(template))
    peaks, properties = find_peaks(mean_correlation, height=threshold, distance=distance)
    return postprocess(zip(peaks, properties['peak_heights']), correlations, data, template, travel_times,
                       template_magnitude, correlation_dmad, tolerance=tolerance,
                       magnitude_threshold=magnitude_threshold)


def preprocess(whole_data: Stream, whole_template: Stream, all_travel_times: Dict[str, float],
               max_channels: int = 16, cc_min_std_factor: float = 0.25, cc_max_std_factor: float = 1.5,
               mapf: Callable = map) -> Tuple[Stream, Stream, Stream, Dict[str, float]]:
    data, template, travel_times = match_traces(whole_data, whole_template, all_travel_times, max_channels)
    correlations = Stream(traces=mapf(correlate_trace, data, template, travel_times.values()))
    stds = np.fromiter(mapf(lambda trace: bn.nanstd(np.abs(trace.data)), correlations), dtype=float)
    mean_std = bn.nanmean(stds)
    traces = zip(correlations, data, template, list(travel_times))
    for std, (xcor_trace, cont_trace, temp_trace, trace_id) in zip(stds, traces):
        if not cc_min_std_factor * mean_std < std < cc_max_std_factor * mean_std:
            logging.debug(f"Ignored trace {xcor_trace} with std {std} (mean: {mean_std})")
            correlations.remove(xcor_trace)
            data.remove(cont_trace)
            template.remove(temp_trace)
            del travel_times[trace_id]
    return correlations, data, template, travel_times


def match_traces(data: Stream, template: Stream, travel_times: Dict[str, float],
                 max_channels: int) -> Tuple[Stream, Stream, Dict[str, float]]:
    trace_ids = sorted(set.intersection({trace.id for trace in data},
                                        {trace.id for trace in template},
                                        set(travel_times)),
                       key=lambda trace_id: (travel_times[trace_id], trace_id[-1]))[:max_channels]
    logging.debug(f"Traces used: {', '.join(trace_ids)}")
    data = select_traces(data, trace_ids)
    template = select_traces(template, trace_ids)
    travel_times = OrderedDict([(trace_id, travel_times[trace_id]) for trace_id in trace_ids])
    return data, template, travel_times


def select_traces(stream: Stream, trace_ids: Iterable[str]) -> Stream:
    def safe_select(trace_id: str):
        for trace in stream:
            if trace_id == trace.id:
                return trace

    return Stream(traces=map(safe_select, trace_ids))


def correlate_trace(continuous: Trace, template: Trace, delay: float) -> Trace:
    header = {"network": continuous.stats.network,
              "station": continuous.stats.station,
              "channel": continuous.stats.channel,
              "starttime": continuous.stats.starttime,
              "sampling_rate": continuous.stats.sampling_rate}
    trace = Trace(data=correlate_data(continuous.data, template.data), header=header)

    duration = continuous.stats.endtime - continuous.stats.starttime
    starttime = trace.stats.starttime + delay
    endtime = starttime + duration
    trace.trim(starttime=starttime, endtime=endtime, nearest_sample=True, pad=True, fill_value=0)
    return trace


def correlate_data(data: np.ndarray, template: np.ndarray) -> np.ndarray:
    template = template - bn.nanmean(template)
    template_length = len(template)
    cross_correlation = np.correlate(data, template, mode='valid')
    pad = len(cross_correlation) - (len(data) - template_length)
    pad1, pad2 = (pad + 1) // 2, pad // 2
    data = np.hstack([np.zeros(pad1), data, np.zeros(pad2)])
    norm = np.sqrt(template_length * bn.move_var(data, template_length)[template_length:] * bn.ss(template))
    mask = norm > np.finfo(cross_correlation.dtype).eps
    np.divide(cross_correlation, norm, where=mask, out=cross_correlation)
    cross_correlation[~mask] = 0
    return cross_correlation


def postprocess(peaks, correlations: Stream, data: Stream, template: Stream, travel_times: Dict[str, float],
                template_magnitude: float, correlation_dmad: float, tolerance: int = 6,
                magnitude_threshold: float = 2.0) -> Generator[Event, None, None]:
    correlations_starttime = min(trace.stats.starttime for trace in correlations)
    correlation_delta = sum(trace.stats.delta for trace in correlations) / len(correlations)
    travel_starttime = min(travel_times.values())
    template_starttime = min(trace.stats.starttime for trace in template)
    for peak, peak_height in peaks:
        trigger_time = correlations_starttime + peak * correlation_delta
        event_date = trigger_time + travel_starttime
        channels = [fix_correlation(trace, peak, tolerance) for trace in correlations]
        mean_correlation = sum(corr for _, corr, _ in channels) / len(channels)
        magnitude = estimate_magnitude(data, template, template_magnitude, trigger_time - template_starttime,
                                       threshold_factor=magnitude_threshold)
        yield event_date.datetime, magnitude, mean_correlation, peak_height, correlation_dmad, channels


def fix_correlation(trace: Trace, trigger_sample: int, tolerance: int) -> CorrelationFix:
    lower = max(trigger_sample - tolerance, 0)
    upper = min(trigger_sample + tolerance + 1, len(trace.data))
    sample_shift = bn.nanargmax(trace.data[lower:upper]) - tolerance
    correlation = trace.data[trigger_sample + sample_shift]
    return trace.id, correlation, sample_shift


def estimate_magnitude(data: Stream, template: Stream, template_magnitude: float, delta: datetime.timedelta,
                       threshold_factor: float) -> float:
    channels_magnitude = []
    for template_trace, data_trace in zip(template, data):
        starttime = template_trace.stats.starttime + delta
        endtime = starttime + (template_trace.stats.endtime - template_trace.stats.starttime)
        continuous_trace_view = data_trace.slice(starttime=starttime, endtime=endtime)
        continuous_max_amp = bn.nanmax(np.abs(continuous_trace_view.data))
        template_max_amp = bn.nanmax(np.abs(template_trace.data))
        if template_max_amp > 0 and continuous_max_amp > 0:
            channel_magnitude = template_magnitude - log10(template_max_amp / continuous_max_amp)
        else:
            channel_magnitude = np.nan
        channels_magnitude.append(channel_magnitude)
    channels_magnitude = np.asarray(channels_magnitude)
    magnitude_deviations = np.abs(channels_magnitude - bn.nanmedian(channels_magnitude))
    magnitude_mad = bn.nanmedian(magnitude_deviations)
    threshold = threshold_factor * magnitude_mad + np.finfo(magnitude_mad).eps
    return bn.nanmean(channels_magnitude[magnitude_deviations < threshold])


def save_records(events: List[Event], output: Path, cc_threshold: float = 0.35, min_channels: int = 6) -> None:
    records = []
    for event in events:
        num_channels = sum(1 for _, cc, _ in event[-1] if cc > cc_threshold)
        if num_channels >= min_channels:
            records.append(event[:-1] + (num_channels,))
    events_dataframe = pd.DataFrame.from_records(records, columns=['template', 'date', 'magnitude', 'correlation',
                                                                   'stack_height', 'stack_dmad', 'num_channels'])
    events_dataframe.sort_values(by=['template', 'date'], inplace=True)
    events_dataframe['crt_pre'] = events_dataframe['stack_height'] / events_dataframe['stack_dmad']
    events_dataframe['crt_post'] = events_dataframe['correlation'] / events_dataframe['stack_dmad']
    logging.info(f"Writing outputs to {output}")
    events_dataframe.to_csv(output, index=False, header=False, na_rep='NA', sep=' ',
                            date_format='%Y-%m-%dT%H:%M:%S.%fZ',
                            float_format='%.3f', columns=['template', 'date', 'magnitude', 'correlation', 'crt_post',
                                                          'stack_height', 'crt_pre', 'num_channels'])