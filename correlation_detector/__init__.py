import logging
import os
import re
from collections import OrderedDict
from concurrent.futures import as_completed, Executor
from math import log10, nan
from pathlib import Path
from typing import Tuple, Dict, Generator, Iterator

import bottleneck as bn
import numpy as np
import pandas as pd
from obspy import read, Stream, Trace, UTCDateTime
from psutil import Process

try:
    import cupy
except ImportError:
    cupy = None


def read_data(path: Path, freqmin: float = 3.0, freqmax: float = 8.0) -> Stream:
    logging.info(f"Reading continuous data from {path}")
    with path.open('rb') as file:
        data = read(file, dtype=np.float32)
    data.merge(method=1, fill_value=0)
    data.detrend("constant")
    data.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)
    starttime = min(trace.stats.starttime for trace in data)
    endtime = max(trace.stats.endtime for trace in data)
    data.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
    return data


def read_templates(templates_directory: Path,
                   ttimes_directory: Path) -> Generator[Tuple[int, Stream, Dict], None, None]:
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
                yield template_number, template_stream, travel_times
            except OSError as err:
                logging.warning(f"{err} occurred while reading template {template_number}")


def filter_data(stds: np.ndarray, correlations: Stream, data: Stream, template: Stream, travel_times: Dict[str, float],
                mad_factor: float = 10.0) -> None:
    deviations = np.abs(stds - bn.nanmedian(stds))
    mad = bn.nanmedian(deviations)
    std_mean = bn.nanmean(stds)
    std_std = bn.nanstd(stds)
    threshold = mad_factor * mad + np.finfo(mad).eps
    tuples = zip(stds, deviations, correlations, data, template, list(travel_times.keys()))
    for std, dev, xcor_trace, cont_trace, temp_trace, ttimes_id in tuples:
        if dev > threshold:
            logging.debug(f"Skipping {xcor_trace} "
                          f"(correlation std: {std}, stream average: {std_mean} ± {3 * std_std})")
            correlations.remove(xcor_trace)
            data.remove(cont_trace)
            template.remove(temp_trace)
            del travel_times[ttimes_id]


def match_traces(data: Stream, template: Stream, travel_times: Dict[str, float],
                 max_channels: int) -> Tuple[Stream, Stream, Dict[str, float]]:
    trace_ids = sorted(set.intersection({trace.id for trace in data},
                                        {trace.id for trace in template},
                                        set(travel_times)),
                       key=lambda trace_id: (travel_times[trace_id], trace_id[-1]))[:max_channels]
    logging.debug(f"Traces used: {', '.join(trace_ids)}")
    data = Stream(traces=[find_trace(data, trace_id) for trace_id in trace_ids])
    template = Stream(traces=[find_trace(template, trace_id) for trace_id in trace_ids])
    travel_times = OrderedDict([(trace_id, travel_times[trace_id]) for trace_id in trace_ids])
    return data, template, travel_times


def find_trace(stream: Stream, trace_id: str):
    for trace in stream:
        if trace_id == trace.id:
            return trace


def correlate_trace(continuous: Trace, template: Trace, delay: float, stream=None) -> Trace:
    header = {"network": continuous.stats.network,
              "station": continuous.stats.station,
              "channel": continuous.stats.channel,
              "starttime": continuous.stats.starttime,
              "sampling_rate": continuous.stats.sampling_rate}
    trace = Trace(data=correlate_data(continuous.data, template.data, stream), header=header)

    duration = continuous.stats.endtime - continuous.stats.starttime
    starttime = trace.stats.starttime + delay
    endtime = starttime + duration
    trace.trim(starttime=starttime, endtime=endtime, nearest_sample=True, pad=True, fill_value=0)
    return trace


def correlate_data(data: np.ndarray, template: np.ndarray, stream) -> np.ndarray:
    template = template - np.mean(template)
    pad = template.size - 1
    data_mean = np.empty_like(data)
    data_mean[:-pad] = bn.move_mean(data, template.size)[pad:]
    data_mean[-pad:] = data[-pad:]
    data = data - data_mean
    cross_correlation = np.empty_like(data)
    cross_correlation[:-pad] = correlate(data, template, stream)
    cross_correlation[-pad:] = 0.0
    data_std = np.empty_like(data)
    data_std[:-pad] = bn.move_std(data, template.size)[pad:]
    data_std[-pad:] = 1.0
    norm = template.size * np.std(template) * data_std
    mask = norm != 0.0
    cross_correlation[~mask] = 0.0
    np.divide(cross_correlation, norm, where=mask, out=cross_correlation)
    return cross_correlation


if cupy:
    # noinspection PyUnresolvedReferences
    def correlate(data, template, stream):
        with stream:
            cross_correlation = cupy.correlate(cupy.asarray(data), cupy.asarray(template), mode='valid')
            cross_correlation = cupy.asnumpy(cross_correlation, stream=stream)
        return cross_correlation
else:
    def correlate(data, template, stream):
        return np.correlate(data, template, mode='valid')


def max_filter(data, pixels):
    data = np.hstack([np.full(pixels, -1.0), data, np.full(pixels, -1.0)])
    return bn.move_max(data, 2 * pixels + 1)[2 * pixels:]


def filter_peaks(peaks, correlations, threshold, factor):
    for peak in peaks:
        xcs = np.fromiter((trace.data[peak] for trace in correlations), dtype=float)
        deviations = np.abs(xcs - np.median(xcs))
        if np.mean(xcs[deviations < factor * np.median(deviations)]) > threshold:
            yield peak


def process_detections(detections: Iterator[int], correlations: Stream, data: Stream, template: Stream,
                       travel_times: Dict[str, float], pool: Executor, tolerance: int) -> Generator[Dict, None, None]:
    correlations_starttime = min(trace.stats.starttime for trace in correlations)
    correlation_delta = sum(trace.stats.delta for trace in correlations) / len(correlations)
    travel_starttime = min(travel_times.values())
    template_starttime = min(trace.stats.starttime for trace in template)

    def process_detection(peak):
        trigger_time = correlations_starttime + peak * correlation_delta
        event_date = trigger_time + travel_starttime
        delta = trigger_time - template_starttime
        channels = []
        for correlation_trace, data_trace, template_trace in zip(correlations, data, template):
            magnitude = relative_magnitude(data_trace, template_trace, delta)
            height, correlation, shift = fix_correlation(correlation_trace, peak, tolerance)
            channels.append({'id': correlation_trace.id, 'height': height, 'correlation': correlation, 'shift': shift,
                             'magnitude': magnitude})
        return {'timestamp': event_date.timestamp, 'channels': channels}

    for future in as_completed(pool.submit(process_detection, peak) for peak in detections):
        yield future.result()


def fix_correlation(trace: Trace, peak: int, tolerance: int) -> Tuple[float, float, int]:
    lower = max(peak - tolerance, 0)
    upper = min(peak + tolerance + 1, len(trace.data))
    shift = bn.nanargmax(trace.data[lower:upper]) - tolerance
    height = trace.data[peak]
    correlation = trace.data[peak + shift]
    return height, correlation, shift


def relative_magnitude(data_trace: Trace, template_trace: Trace, delta: float) -> float:
    duration = template_trace.stats.endtime - template_trace.stats.starttime
    starttime = template_trace.stats.starttime + delta
    endtime = starttime + duration
    data_trace_view = data_trace.slice(starttime=starttime, endtime=endtime)
    if data_trace_view:
        data_amp = bn.nanmax(np.abs(data_trace_view.data))
        template_amp = bn.nanmax(np.abs(template_trace.data))
        if (ratio := data_amp / template_amp) > 0:
            return log10(ratio)
        else:
            return nan
    else:
        return nan


def estimate_magnitude(template_magnitude, relative_magnitudes, mad_factor: float) -> float:
    magnitudes = template_magnitude + np.fromiter(relative_magnitudes, dtype=float)
    deviations = np.abs(magnitudes - bn.nanmedian(magnitudes))
    mad = bn.nanmedian(deviations)
    threshold = mad_factor * mad + np.finfo(mad).eps
    return bn.nanmean(magnitudes[deviations < threshold])


def preprocess(detections, catalog, threshold: float, min_channels: int, mag_relative_threshold: float):
    for detection in detections:
        channels = detection['channels']
        num_channels = sum(1 for channel in channels if channel['correlation'] > threshold)
        if num_channels >= min_channels:
            timestamp = UTCDateTime(detection['timestamp'])
            height = sum(channel['height'] for channel in channels) / len(channels)
            correlation = sum(channel['correlation'] for channel in channels) / len(channels)
            template_magnitude = catalog.iloc[detection['template'] - 1]
            magnitude = estimate_magnitude(template_magnitude, [channel['magnitude'] for channel in channels],
                                           mag_relative_threshold)
            detection.update({'datetime': timestamp, 'height': height, 'correlation': correlation,
                              'magnitude': magnitude, 'template_magnitude': template_magnitude,
                              'channels': channels, 'num_channels': num_channels,
                              'crt_pre': height / detection['dmad'], 'crt_post': correlation / detection['dmad']})
            for name, ref in [('30%', 0.3), ('50%', 0.5), ('70%', 0.7), ('90%', 0.9)]:
                detection[name] = sum(1 for channel in channels if channel['correlation'] > ref)
            yield detection


def read_zmap(catalog_path):
    logging.info(f"Reading catalog from {catalog_path}")
    template_magnitudes = pd.read_csv(catalog_path, sep=r'\s+', usecols=(5,), squeeze=True, dtype=float)
    return template_magnitudes


def format_stats(event):
    line = ""
    for trace in event['channels']:
        network, station, _, channel = trace['id'].split('.')
        height, correlation, shift = trace['height'], trace['correlation'], trace['shift']
        line += f"{network}.{station} {channel} {height:.12f} {correlation:.12f} {shift} " \
                f"{event['template_magnitude'] + trace['magnitude']:.12f}\n"
    line += f"{event['datetime'].strftime('%y%m%d')} {event['template']} ? {event['datetime'].isoformat()} " \
            f"{event['magnitude']:.2f} {event['template_magnitude']:.2f} {event['num_channels']} {event['dmad']:.3f} " \
            f"{event['correlation']:.3f} {event['crt_post']:.3f} {event['height']:.3f} {event['crt_pre']:.3f} " \
            f"{event['30%']} {event['50%']} {event['70%']} {event['90%']}\n"

    return line


def format_cat(event):
    return f"{event['template']} {event['datetime'].isoformat()} {event['magnitude']:.2f} " \
           f"{event['correlation']:.3f} {event['crt_post']:.3f} {event['height']:.3f} {event['crt_pre']:.3f} " \
           f"{event['num_channels']}\n"


def memory_usage():
    return Process(os.getpid()).memory_info().rss / (1 << 30)
