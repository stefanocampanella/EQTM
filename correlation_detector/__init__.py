import logging
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

try:
    import cupy
except ImportError:
    cupy = None


def read_data(path: Path, freqmin: float = 3.0, freqmax: float = 8.0) -> Stream:
    logging.info(f"Reading continuous data from {path}")
    with path.open('rb') as file:
        data = read(file, dtype=np.float32)
    data.merge(fill_value=0)
    data.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)
    starttime = min(trace.stats.starttime for trace in data)
    endtime = max(trace.stats.endtime for trace in data)
    data.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
    return data


def read_templates(templates_directory: Path,
                   ttimes_directory: Path) -> Generator[Tuple[int, Stream, Dict], None, None]:
    # logging.info(f"Reading catalog from {catalog_path}")
    # template_magnitudes = pd.read_csv(catalog_path, sep=r'\s+', usecols=(5,), squeeze=True, dtype=float)
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
                template_stream.merge(fill_value=0)
                yield template_number, template_stream, travel_times  # , template_magnitudes.iloc[template_number - 1]
            except OSError as err:
                logging.warning(f"{err} occurred while reading template {template_number}")


def filter_data(stds: np.ndarray, correlations: Stream, data: Stream, template: Stream, travel_times: Dict[str, float],
                min_std: float = 0.25, max_std: float = 1.5) -> None:
    mean_std = bn.nanmean(stds)
    traces = zip(correlations, data, template, list(travel_times))
    for std, (xcor_trace, cont_trace, temp_trace, trace_id) in zip(stds, traces):
        if not min_std * mean_std < std < max_std * mean_std:
            logging.debug(f"Ignored trace {xcor_trace} with std {std} (mean: {mean_std})")
            correlations.remove(xcor_trace)
            data.remove(cont_trace)
            template.remove(temp_trace)
            del travel_times[trace_id]


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
    template = template - bn.nanmean(template)
    template_length = len(template)
    cross_correlation = correlate(data, template, stream)
    pad = len(cross_correlation) - (len(data) - template_length)
    pad1, pad2 = (pad + 1) // 2, pad // 2
    data = np.hstack([np.zeros(pad1), data, np.zeros(pad2)])
    norm = np.sqrt(template_length * bn.move_var(data, template_length)[template_length:] * bn.ss(template))
    mask = norm > np.finfo(cross_correlation.dtype).eps
    np.divide(cross_correlation, norm, where=mask, out=cross_correlation)
    cross_correlation[~mask] = 0
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


def get_detections(peaks: Iterator[int], correlations: Stream, data: Stream, template: Stream,
                   travel_times: Dict[str, float], pool: Executor, tolerance: int = 6) -> Generator[Dict, None, None]:
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

    for future in as_completed(pool.submit(process_detection, peak) for peak in peaks):
        yield future.result()


def fix_correlation(trace: Trace, trigger_sample: int, tolerance: int) -> Tuple[float, float, int]:
    lower = max(trigger_sample - tolerance, 0)
    upper = min(trigger_sample + tolerance + 1, len(trace.data))
    sample_shift = bn.nanargmax(trace.data[lower:upper]) - tolerance
    correlation = trace.data[trigger_sample]
    max_correlation = trace.data[trigger_sample + sample_shift]
    return correlation, max_correlation, sample_shift


def relative_magnitude(data_trace, template_trace, delta):
    duration = template_trace.stats.endtime - template_trace.stats.starttime
    starttime = template_trace.stats.starttime + delta
    endtime = starttime + duration
    data_trace_view = data_trace.slice(starttime=starttime, endtime=endtime)
    data_amp = bn.nanmax(np.abs(data_trace_view.data))
    template_amp = bn.nanmax(np.abs(template_trace.data))
    if (ratio := data_amp / template_amp) > 0:
        return log10(ratio)
    else:
        return nan


def estimate_magnitude(template_magnitude, relative_magnitudes, mad_factor: float) -> float:
    magnitudes = template_magnitude + np.fromiter(relative_magnitudes, dtype=float)
    deviations = np.abs(magnitudes - bn.nanmedian(magnitudes))
    mad = bn.nanmedian(deviations)
    threshold = mad_factor * mad + np.finfo(mad).eps
    return bn.nanmean(magnitudes[deviations < threshold])


def preprocess(detections, catalog, threshold: float = 0.35, min_channels: int = 6, mag_relative_threshold=2.0):
    for detection in detections:
        channels = detection['channels']
        template_magnitude = catalog.iloc[detection['template'] - 1]
        magnitude = estimate_magnitude(template_magnitude, [channel['magnitude'] for channel in channels],
                                       mag_relative_threshold)
        num_channels = sum(1 for channel in channels if channel['correlation'] > threshold)
        correlation = sum(channel['correlation'] for channel in channels) / len(channels)
        if num_channels >= min_channels:
            timestamp = UTCDateTime(detection['timestamp'])
            detection.update({'datetime': timestamp, 'num_channels': num_channels, 'correlation': correlation,
                              'magnitude': magnitude, 'template_magnitude': template_magnitude})
            for name, ref in [('30%', 0.3), ('50%', 0.5), ('70%', 0.7), ('90%', 0.9)]:
                detection[name] = sum(1 for channel in channels if channel['correlation'] > ref)
            detection['crt_pre'] = detection['height'] / detection['dmad']
            detection['crt_post'] = detection['correlation'] / detection['dmad']
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
        line += f"{network}.{station} {channel} {height:.12f} {correlation:.12f} {shift}\n"
    line += f"{event['datetime'].strftime('%y%m%d')} {event['template']} ? {event['datetime'].isoformat()} " \
            f"{event['magnitude']:.2f} {event['template_magnitude']:.2f} {event['num_channels']} {event['dmad']:.3f} " \
            f"{event['correlation']:.3f} {event['crt_post']:.3f} {event['height']:.3f} {event['crt_pre']:.3f} " \
            f"{event['30%']} {event['50%']} {event['70%']} {event['90%']}\n"

    return line


def format_cat(event):
    return f"{event['template']} {event['datetime'].isoformat()} {event['magnitude']:.2f} " \
           f"{event['correlation']:.3f} {event['crt_post']:.3f} {event['height']:.3f} {event['crt_pre']:.3f} " \
           f"{event['num_channels']}\n"
