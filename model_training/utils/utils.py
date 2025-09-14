import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import seaborn as sns
from scipy import signal

def visualize_eeg_channels(df: pd.DataFrame, channels: List[str],
                          start_time: float = 0, duration: float = 10,
                          sample_rate: int = 250):
    samples_to_plot = int(duration * sample_rate)
    start_sample = int(start_time * sample_rate)
    end_sample = min(start_sample + samples_to_plot, len(df))

    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 2*len(channels)),
                            sharex=True)
    if len(channels) == 1:
        axes = [axes]

    time_axis = np.arange(start_sample, end_sample) / sample_rate

    for idx, channel in enumerate(channels):
        if channel in df.columns:
            axes[idx].plot(time_axis, df[channel].iloc[start_sample:end_sample])
            axes[idx].set_ylabel(channel)
            axes[idx].grid(True, alpha=0.3)

            if 'Marker' in df.columns:
                marker_data = df['Marker'].iloc[start_sample:end_sample]
                touch_indices = np.where(marker_data > 0)[0]
                if len(touch_indices) > 0:
                    axes[idx].scatter(time_axis[touch_indices],
                                    df[channel].iloc[start_sample:end_sample].iloc[touch_indices],
                                    c='red', s=10, alpha=0.5, label='Touch')
                    axes[idx].legend()

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'EEG Channels ({start_time:.1f}s - {start_time+duration:.1f}s)')
    plt.tight_layout()
    return fig

def plot_frequency_spectrum(signal_data: np.ndarray, sample_rate: int = 250,
                           title: str = "Frequency Spectrum"):
    freqs, psd = signal.periodogram(signal_data, fs=sample_rate)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(freqs, psd)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 60])

    band_colors = {
        'Delta (0.5-4 Hz)': (0.5, 4, 'blue'),
        'Theta (4-8 Hz)': (4, 8, 'green'),
        'Alpha (8-13 Hz)': (8, 13, 'yellow'),
        'Beta (13-30 Hz)': (13, 30, 'orange'),
        'Gamma (30-50 Hz)': (30, 50, 'red')
    }

    for band_name, (low, high, color) in band_colors.items():
        ax.axvspan(low, high, alpha=0.2, color=color, label=band_name)

    ax.legend()
    plt.tight_layout()
    return fig

def create_windowed_labels(markers: np.ndarray, window_size: int = 250,
                          overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    step_size = int(window_size * (1 - overlap))
    window_labels = []
    window_indices = []

    for start in range(0, len(markers) - window_size + 1, step_size):
        end = start + window_size
        window_marker = markers[start:end]
        label = 1 if np.any(window_marker > 0) else 0
        window_labels.append(label)
        window_indices.append((start, end))

    return np.array(window_labels), np.array(window_indices)

def plot_model_comparison(results_dict: Dict[str, Dict], metric: str = 'accuracy'):
    models = list(results_dict.keys())
    scores = [results_dict[model][metric] for model in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, scores)

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')

    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'Model Comparison - {metric.capitalize()}')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig

def analyze_touch_events(df: pd.DataFrame, sample_rate: int = 250) -> Dict[str, any]:
    markers = df['Marker'].values if 'Marker' in df.columns else None
    if markers is None:
        return {}

    touch_starts = []
    touch_ends = []
    in_touch = False

    for i in range(len(markers)):
        if markers[i] > 0 and not in_touch:
            touch_starts.append(i)
            in_touch = True
        elif markers[i] == 0 and in_touch:
            touch_ends.append(i)
            in_touch = False

    if in_touch:
        touch_ends.append(len(markers) - 1)

    durations = [(end - start) / sample_rate for start, end in zip(touch_starts, touch_ends)]

    analysis = {
        'total_touches': len(touch_starts),
        'avg_duration': np.mean(durations) if durations else 0,
        'std_duration': np.std(durations) if durations else 0,
        'min_duration': np.min(durations) if durations else 0,
        'max_duration': np.max(durations) if durations else 0,
        'touch_starts': touch_starts,
        'touch_ends': touch_ends,
        'durations': durations
    }

    return analysis

def plot_touch_distribution(analysis: Dict[str, any]):
    if not analysis or 'durations' not in analysis:
        print("No touch events found in data")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(analysis['durations'], bins=20, edgecolor='black')
    axes[0].set_xlabel('Duration (seconds)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Touch Duration Distribution')
    axes[0].axvline(analysis['avg_duration'], color='red',
                   linestyle='--', label=f"Mean: {analysis['avg_duration']:.2f}s")
    axes[0].legend()

    touch_times = [(start + end) / 2 for start, end in
                  zip(analysis['touch_starts'], analysis['touch_ends'])]
    axes[1].scatter(touch_times, analysis['durations'])
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Duration (seconds)')
    axes[1].set_title('Touch Events Timeline')
    axes[1].axhline(analysis['avg_duration'], color='red',
                   linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig

def validate_data_quality(df: pd.DataFrame, channels: List[str]) -> Dict[str, any]:
    quality_report = {}

    for channel in channels:
        if channel not in df.columns:
            quality_report[channel] = {'status': 'missing'}
            continue

        data = df[channel].values
        quality_report[channel] = {
            'status': 'present',
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'zeros': np.sum(data == 0),
            'inf_values': np.sum(np.isinf(data)),
            'nan_values': np.sum(np.isnan(data)),
            'saturation': np.sum(np.abs(data) > np.percentile(np.abs(data), 99.9))
        }

    missing_channels = [ch for ch, info in quality_report.items()
                       if info.get('status') == 'missing']
    problematic_channels = [ch for ch, info in quality_report.items()
                           if info.get('status') == 'present' and
                           (info['inf_values'] > 0 or info['nan_values'] > 0)]

    summary = {
        'total_channels': len(channels),
        'present_channels': len(channels) - len(missing_channels),
        'missing_channels': missing_channels,
        'problematic_channels': problematic_channels,
        'channel_details': quality_report
    }

    return summary

def print_quality_report(quality_report: Dict[str, any]):
    print("="*50)
    print("Data Quality Report")
    print("="*50)
    print(f"Total channels expected: {quality_report['total_channels']}")
    print(f"Channels present: {quality_report['present_channels']}")

    if quality_report['missing_channels']:
        print(f"\nMissing channels: {', '.join(quality_report['missing_channels'])}")

    if quality_report['problematic_channels']:
        print(f"\nProblematic channels: {', '.join(quality_report['problematic_channels'])}")

    print("\nChannel Statistics:")
    for channel, info in quality_report['channel_details'].items():
        if info.get('status') == 'present':
            print(f"\n{channel}:")
            print(f"  Mean: {info['mean']:.2f}, Std: {info['std']:.2f}")
            print(f"  Range: [{info['min']:.2f}, {info['max']:.2f}]")
            if info['nan_values'] > 0:
                print(f"  WARNING: {info['nan_values']} NaN values")
            if info['inf_values'] > 0:
                print(f"  WARNING: {info['inf_values']} Inf values")

def main():
    from preprocess import EEGPreprocessor

    print("Testing utility functions...")
    preprocessor = EEGPreprocessor(sample_rate=250)
    df = preprocessor.load_data("OpenBCI-RAW-2025-09-13_23-58-04.csv")

    eeg_channels = ['C3', 'C4', 'P3', 'P4', 'P7', 'P8', 'T7', 'T8']
    quality_report = validate_data_quality(df, eeg_channels)
    print_quality_report(quality_report)

    touch_analysis = analyze_touch_events(df, sample_rate=250)
    if touch_analysis:
        print(f"\nTouch Event Analysis:")
        print(f"  Total touches: {touch_analysis['total_touches']}")
        print(f"  Average duration: {touch_analysis['avg_duration']:.2f}s")
        print(f"  Duration range: [{touch_analysis['min_duration']:.2f}s, "
              f"{touch_analysis['max_duration']:.2f}s]")

    print("\nGenerating visualizations...")
    fig1 = visualize_eeg_channels(df, ['C3', 'C4', 'P3', 'P4'],
                                 start_time=10, duration=5)
    plt.savefig('eeg_channels_sample.png')

    if 'C3' in df.columns:
        fig2 = plot_frequency_spectrum(df['C3'].values[:2500], sample_rate=250,
                                      title="C3 Channel Frequency Spectrum")
        plt.savefig('frequency_spectrum_c3.png')

    if touch_analysis:
        fig3 = plot_touch_distribution(touch_analysis)
        if fig3:
            plt.savefig('touch_distribution.png')

    print("Visualizations saved!")

if __name__ == "__main__":
    main()