import logging
import re
from collections import OrderedDict
from contextlib import nullcontext
from math import log10, nan, inf
from pathlib import Path
from typing import Tuple, Dict, Generator, Iterator

import bottleneck as bn
import numpy as np
import pandas as pd
from obspy import read, Stream, Trace, UTCDateTime

try:
    import cupy

    xp = cupy
except ImportError:
    cupy = None
    xp = np


def read_data(path: Path, freqmin: float = 3.0, freqmax: float = 8.0) -> Stream:
    logging.info(f"Reading continuous data from {path}")
    with path.open('rb') as file:
        data = read(file, dtype=np.float64)
    data.merge(fill_value=0)
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
                    template_stream = read(template_file, dtype=np.float64)
                template_stream.merge(fill_value=0)
                yield template_number, template_stream, travel_times
            except OSError as err:
                logging.warning(f"{err} occurred while reading template {template_number}")


def filter_data(stds: np.ndarray, correlations: Stream, data: Stream, template: Stream, travel_times: Dict[str, float],
                mad_factor: float = 10.0) -> None:
    deviations = np.abs(stds - np.median(stds))
    mad = np.median(deviations)
    std_mean = np.mean(stds)
    std_std = np.std(stds)
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


def correlate_trace(continuous: Trace, template: Trace, delay: float, stream=nullcontext()) -> Trace:
    header = {"network": continuous.stats.network,
              "station": continuous.stats.station,
              "channel": continuous.stats.channel,
              "starttime": continuous.stats.starttime,
              "sampling_rate": continuous.stats.sampling_rate}
    with stream:
        correlation = correlate_data(continuous.data, template.data)
    trace = Trace(data=correlation, header=header)

    duration = continuous.stats.endtime - continuous.stats.starttime
    starttime = trace.stats.starttime + delay
    endtime = starttime + duration
    trace.trim(starttime=starttime, endtime=endtime, nearest_sample=True, pad=True, fill_value=0)
    return trace


def correlate_data(data: np.ndarray, template: np.ndarray) -> np.ndarray:
    data = xp.asarray(data)
    template = xp.asarray(template)
    template -= xp.mean(template)
    pad = template.size - 1
    correlation = xp.empty_like(data)
    correlation[:-pad] = xp.correlate(data, template, mode='valid')
    correlation[-pad:] = 0.0
    norm = moving_mean(data * data, template.size)
    mean_squared = moving_mean(data, template.size)
    mean_squared *= mean_squared
    norm -= mean_squared
    mask = norm <= 0.0
    norm *= template.size * xp.dot(template, template)
    norm[mask] = 1.0
    xp.sqrt(norm, out=norm)
    correlation[mask] = 0.0
    correlation /= norm
    if xp == cupy:
        # noinspection PyUnresolvedReferences
        return cupy.asnumpy(correlation, stream=cupy.cuda.get_current_stream())
    else:
        return correlation


def moving_mean(data, window):
    pad = window - 1
    mean = xp.empty_like(data)
    csum = xp.cumsum(data)
    mean[0] = csum[pad] / window
    mean[1:-pad] = (csum[window:] - csum[:-window]) / window
    mean[-pad:] = 0.0
    return mean


def max_filter(data, pixels):
    data = np.hstack([np.full(pixels, -1.0), data, np.full(pixels, -1.0)])
    return bn.move_max(data, 2 * pixels + 1)[2 * pixels:]


def filter_peaks(peaks, shifted_correlations, factor, threshold):
    for peak in peaks:
        peak_correlations = np.fromiter((trace.data[peak] for trace in shifted_correlations), dtype=float)
        deviations = np.abs(peak_correlations - np.median(peak_correlations))
        mad = np.mean(deviations)
        valid_correlations = peak_correlations[deviations < factor * mad]
        if np.mean(valid_correlations) > threshold:
            yield peak


def process_detections(peaks: Iterator[int], correlations: Stream, data: Stream, template: Stream,
                       travel_times: Dict[str, float], tolerance: int) -> Generator[Dict, None, None]:
    correlations_starttime = min(trace.stats.starttime for trace in correlations)
    correlation_delta = sum(trace.stats.delta for trace in correlations) / len(correlations)
    travel_starttime = min(travel_times.values())
    template_starttime = min(trace.stats.starttime for trace in template)

    for peak in peaks:
        trigger_time = correlations_starttime + peak * correlation_delta
        event_date = trigger_time + travel_starttime
        delta = trigger_time - template_starttime
        channels = []
        for correlation_trace, data_trace, template_trace in zip(correlations, data, template):
            magnitude = relative_magnitude(data_trace, template_trace, delta)
            height, correlation, shift = fix_correlation(correlation_trace, peak, tolerance)
            channels.append({'id': correlation_trace.id, 'height': height, 'correlation': correlation, 'shift': shift,
                             'magnitude': magnitude})
        yield {'timestamp': event_date.timestamp, 'channels': channels}


def fix_correlation(trace: Trace, peak: int, tolerance: int) -> Tuple[float, float, int]:
    lower = max(peak - tolerance, 0)
    upper = min(peak + tolerance + 1, len(trace.data))
    shift = np.argmax(trace.data[lower:upper]) - tolerance
    height = trace.data[peak]
    correlation = trace.data[peak + shift]
    return height, correlation, shift


def relative_magnitude(data_trace: Trace, template_trace: Trace, delta: float) -> float:
    duration = template_trace.stats.endtime - template_trace.stats.starttime
    starttime = template_trace.stats.starttime + delta
    endtime = starttime + duration
    data_trace_view = data_trace.slice(starttime=starttime, endtime=endtime)
    if data_trace_view:
        data_amp = np.max(np.abs(data_trace_view.data))
        template_amp = np.max(np.abs(template_trace.data))
        if (ratio := data_amp / template_amp) > 0:
            return log10(ratio)
        else:
            return nan
    else:
        return nan


def estimate_magnitude(template_magnitude, relative_magnitudes, mad_factor: float):
    magnitudes = template_magnitude + np.fromiter(relative_magnitudes, dtype=float)
    deviations = np.abs(magnitudes - np.nanmedian(magnitudes))
    mad = np.nanmedian(deviations)
    threshold = mad_factor * mad + np.finfo(mad).eps
    return np.mean(magnitudes[deviations < threshold])


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
            dmad = detection['dmad']
            detection.update({'datetime': timestamp, 'height': height, 'correlation': correlation,
                              'magnitude': magnitude, 'template_magnitude': template_magnitude,
                              'channels': channels, 'num_channels': num_channels,
                              'crt_pre': inf if dmad == 0.0 else height / dmad,
                              'crt_post': inf if dmad == 0.0 else correlation / dmad})
            for name, ref in [('30%', 0.3), ('50%', 0.5), ('70%', 0.7), ('90%', 0.9)]:
                detection[name] = sum(1 for channel in channels if channel['correlation'] > ref)
            yield detection


def flatten(events_buffer):
    for template_data in events_buffer:
        for detection in template_data['detections']:
            detection.update({'template': template_data['template'], 'dmad': template_data['dmad']})
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
