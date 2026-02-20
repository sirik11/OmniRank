"""
streaming_service.py
--------------------

This module simulates a real‑time event stream for the feed ranking platform.  In production
you would connect to a message broker (e.g., Kafka, Pulsar) and emit interaction events
as they happen.  Here we provide a simple generator that yields events from the existing
Parquet file at a configurable rate.

You can integrate this with the rest of the system by consuming events and updating
models or logs in real time.
"""

import time
import argparse
import pandas as pd


def event_stream(events_path: str, rate: float = 100.0):
    """
    Yield events from a Parquet file at a given rate (events per second).
    """
    events = pd.read_parquet(events_path)
    for _, row in events.iterrows():
        yield row.to_dict()
        time.sleep(1.0 / rate)


def main():
    parser = argparse.ArgumentParser(description="Simulate a streaming event source.")
    parser.add_argument('--events_path', type=str, required=True, help='Path to events Parquet file')
    parser.add_argument('--rate', type=float, default=100.0, help='Events per second')
    args = parser.parse_args()
    for event in event_stream(args.events_path, rate=args.rate):
        print(event)  # In production you would send to Kafka or a queue


if __name__ == '__main__':
    main()