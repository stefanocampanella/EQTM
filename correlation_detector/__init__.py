import logging
import re
from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
from functools import lru_cache
from math import log10, nan, inf
from pathlib import Path
from typing import Tuple, Dict, Generator
from datetime import timedelta, datetime, timezone

import bottleneck as bn
import numpy as np
import pandas as pd
from obspy import read, Stream, Trace

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


def read_templatesdata(templates_directory: Path,
                       ttimes_directory: Path) -> Generator[Tuple[int, Stream, Dict], None, None]:
    logging.info(f"Reading travel times from {ttimes_directory}")
    logging.info(f"Reading templates from {templates_directory}")
    file_regex = re.compile(r'(?P<template_number>\d+).mseed')
    for template_path in templates_directory.glob('*.mseed'):
        match = file_regex.match(template_path.name)
        if match:
            template_number = int(match.group('template_number'))
            try:
                travel_times = read_ttimes(ttimes_directory, template_number)
                template_stream = read_template(templates_directory, template_number)
                yield template_number, template_stream, travel_times
            except OSError as exception:
                logging.warning(f"OSError occurred while reading template {template_number}", exc_info=exception)


def read_template(templates_directory, template_number):
    template_path = templates_directory / f"{template_number}.mseed"
    logging.debug(f"Reading from {template_path}")
    with template_path.open('rb') as template_file:
        template_stream = read(template_file, dtype=np.float64)
    template_stream.merge(fill_value=0)
    return template_stream


@lru_cache
def read_ttimes(ttimes_directory, template_number):
    ttimes_path = ttimes_directory / f"{template_number}.ttimes"
    travel_times = OrderedDict()
    logging.debug(f"Reading from {ttimes_path}")
    with open(ttimes_path, "r") as ttimes_file:
        while line := ttimes_file.readline():
            key, value_string = line.split(' ')
            network, station, channel = key.split('.')
            trace_id = f"{network}.{station}..{channel}"
            value = float(value_string)
            travel_times[trace_id] = value
    return travel_times


def match_traces(data: Stream, template: Stream, travel_times: Dict[str, float],
                 max_channels: int) -> Tuple[Stream, Stream, Dict[str, float]]:
    trace_ids = sorted(set.intersection({trace.id for trace in data},
                                        {trace.id for trace in template},
                                        set(travel_times)),
                       key=lambda trace_id: (travel_times[trace_id], trace_id[-1]))[:max_channels]
    if not trace_ids:
        raise RuntimeError("No matches between data, template and travel times")
    logging.debug(f"Traces used: {', '.join(trace_ids)}")
    data = Stream(traces=[find_trace(data, trace_id) for trace_id in trace_ids])
    template = Stream(traces=[find_trace(template, trace_id) for trace_id in trace_ids])
    travel_times = OrderedDict([(trace_id, travel_times[trace_id]) for trace_id in trace_ids])
    if any(data_trace.stats.delta != template_trace.stats.delta for data_trace, template_trace in zip(data, template)):
        raise RuntimeError("Data and template must have the same delta")
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

    starttime = continuous.stats.starttime + delay
    endtime = continuous.stats.endtime + delay
    trace.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
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
    data = np.hstack([np.full(pixels, data[0]), data, np.full(pixels, data[-1])])
    return bn.move_max(data, 2 * pixels + 1)[2 * pixels:]


def max_correlation(trace: Trace, peak: int, tolerance: int) -> Tuple[float, float, int]:
    lower = max(peak - tolerance, 0)
    upper = min(peak + tolerance + 1, len(trace.data))
    shift = np.argmax(trace.data[lower:upper]) - tolerance
    return trace.data[peak], trace.data[peak + shift], shift


def relative_magnitude(data_trace: Trace, template_trace: Trace, delay: float) -> float:
    starttime = template_trace.stats.starttime + delay
    endtime = template_trace.stats.endtime + delay
    data_trace_view = data_trace.slice(starttime=starttime, endtime=endtime)
    if data_trace_view and template_trace:
        data_amp = np.max(np.abs(data_trace_view.data))
        template_amp = np.max(np.abs(template_trace.data))
        if template_amp == 0.0:
            return nan
        elif (ratio := data_amp / template_amp) <= 0.0:
            return nan
        else:
            return log10(ratio)
    else:
        return nan


def estimate_magnitude(template_magnitude, relative_magnitudes, mad_factor: float):
    magnitudes = template_magnitude + np.fromiter(relative_magnitudes, dtype=float)
    deviations = np.abs(magnitudes - np.nanmedian(magnitudes))
    mad = np.nanmedian(deviations)
    threshold = mad_factor * mad + np.finfo(mad).eps
    return np.mean(magnitudes[deviations < threshold])


def flatten(events_buffer):
    for template_data in events_buffer:
        for detection in template_data['detections']:
            detection.update({'template': template_data['template'], 'dmad': template_data['dmad']})
            yield detection


def read_zmap(catalog_path):
    zmap = pd.read_csv(catalog_path,
                       sep=r'\s+',
                       usecols=range(10),
                       names=['longitude',
                              'latitude',
                              'year',
                              'month',
                              'day',
                              'magnitude',
                              'depth',
                              'hour',
                              'minute',
                              'second'],
                       parse_dates={'date': ['year', 'month', 'day', 'hour', 'minute', 'second']},
                       date_parser=lambda datestr: pd.to_datetime(datestr, format='%Y %m %d %H %M %S.%f', utc=True))
    zmap['timestamp'] = zmap.date.map(lambda t: t.timestamp())
    zmap.drop('date', axis=1, inplace=True)
    return zmap


def fix_correlations(event, corr_atol):
    fixed_event = deepcopy(event)
    fixed_event['channels'] = list(filter(lambda channel: channel['correlation'] <= 1 + corr_atol, event['channels']))
    return fixed_event


def make_record(event, zmap, ttimes_directory):
    record = {}
    for key in ["template", "timestamp", "dmad"]:
        record[key] = event[key]

    template = event['template']
    for key in zmap.keys():
        record["template_" + key] = zmap.loc[template, key]

    channels = event['channels']
    for channel in channels:
        channel_name = channel['id']
        record[f"{channel_name}_ttime"] = read_ttimes(ttimes_directory, template)[channel_name]
        for key in channel.keys():
            if key != 'id':
                record[f"{channel_name}_{key}"] = channel[key]
    num_stations = len({station for station, _ in map(lambda ch: ch['id'].split('..'), event['channels'])})
    num_channels = len(event['channels'])
    correlation_mean = sum(ch['correlation'] for ch in event['channels']) / num_channels
    record.update({'num_stations': num_stations,
                   'num_channels': num_channels,
                   'correlation_mean': correlation_mean})
    return record


def make_legacy_record(record, corr_threshold, mag_relative_threshold):
    channels = channels_from_record(record)
    num_channels_above = sum(1 for channel in channels if channel['correlation'] > corr_threshold)
    magnitude = estimate_magnitude(record['template_magnitude'], [channel['magnitude'] for channel in channels],
                                   mag_relative_threshold)
    height = sum(channel['height'] for channel in channels) / len(channels)
    correlation = record['correlation_mean']
    dmad = record['dmad']
    record.update({'height': height, 'correlation': correlation,
                   'magnitude': magnitude, 'channels': channels, 'num_channels_above': num_channels_above,
                   'crt_pre': inf if dmad == 0.0 else height / dmad,
                   'crt_post': inf if dmad == 0.0 else correlation / dmad})
    for name, ref in [('30%', 0.3), ('50%', 0.5), ('70%', 0.7), ('90%', 0.9)]:
        record[name] = sum(1 for channel in channels if channel['correlation'] > ref)

    return record


def channels_from_record(record):
    features = ['magnitude', 'height', 'correlation', 'shift']
    channels = dict()
    for key, value in record.items():
        if any(key.endswith('_' + feature_name) for feature_name in features):
            channel_id, feature = key.split('_')
            if channel_id == 'template':
                continue
            elif channel_id in channels:
                channels[channel_id][feature] = value
            else:
                channels[channel_id] = {feature: value}
    return [channel_data | {'id': channel_id} for channel_id, channel_data in channels.items()
            if np.isfinite(channel_data['correlation'])]


def format_stats(event):
    line = ""
    for trace in event['channels']:
        network, station, _, channel = trace['id'].split('.')
        height, correlation, shift = trace['height'], trace['correlation'], trace['shift']
        line += f"{network}.{station} {channel} {height:.12f} {correlation:.12f} {shift} " \
                f"{event['template_magnitude'] + trace['magnitude']:.12f}\n"
    line += f"{event['date'].strftime('%y%m%d')} {event['template']} ? " \
            f"{event['date'].strftime('%Y-%m-%dT%H:%M:%S.%f')} " \
            f"{event['magnitude']:.2f} {event['template_magnitude']:.2f} " \
            f"{event['num_channels_above']} {event['dmad']:.3f} " \
            f"{event['correlation']:.3f} {event['crt_post']:.3f} {event['height']:.3f} {event['crt_pre']:.3f} " \
            f"{event['30%']} {event['50%']} {event['70%']} {event['90%']}\n"

    return line


def format_cat(event):
    return f"{event['template']} {event['date'].strftime('%Y-%m-%dT%H:%M:%S.%f')} {event['magnitude']:.2f} " \
           f"{event['correlation']:.3f} {event['crt_post']:.3f} {event['height']:.3f} {event['crt_pre']:.3f} " \
           f"{event['num_channels_above']}\n"


def find_records(catalogue, selection, dt):
    try:
        indices = []
        for row in selection.itertuples():
            start, stop = row.date - dt, row.date + dt
            candidates = catalogue[(catalogue.index > start) &
                                   (catalogue.index < stop) &
                                   (catalogue['template'] == row.template)]
            if not candidates.empty:
                indices.append(candidates['correlation_mean'].idxmax())
        if indices:
            return catalogue.loc[indices]
        else:
            logging.debug(f"No events found in {catalogue}")
    except BaseException as exception:
        logging.debug(f"An error occurred while processing {catalogue}", exc_info=exception)


def filenames_from_dates(dates, root=Path('.'), extension='.parquet'):
    filenames = set()
    for date in dates:
        date_string = date.strftime("%Y-%m-%d")
        filename = root / (date_string + extension)
        if filename.exists():
            filenames.add(filename)
        if date.hour == 0:
            daybefore_string = (date - timedelta(days=1)).strftime("%Y-%m-%d")
            filename = root / (daybefore_string + extension)
            if filename.exists():
                filenames.add(filename)
        elif date.hour == 23:
            dayafter_string = (date + timedelta(days=1)).strftime("%Y-%m-%d")
            filename = root / (dayafter_string + extension)
            if filename.exists():
                filenames.add(filename)
    return list(filenames)
