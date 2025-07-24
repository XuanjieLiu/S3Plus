"""
Rule-based pitch classifier for single-note mel-spectrograms.
This doesn't work well.
"""

import torch
import numpy as np
from scipy.signal import find_peaks, medfilt


def extract_f0_from_mel(
    mel_spec, sr=16000, hop_length=256, n_mels=128, f_min=0, f_max=8000
):
    """
    Extract F0 from mel spectrogram using rule-based peak detection

    Args:
        mel_spec: (n_mels, n_frames) mel spectrogram
        sr: sample rate
        hop_length: hop length used in mel transform
        n_mels: number of mel bins
        f_min, f_max: frequency range of mel scale

    Returns:
        f0_contour: (n_frames,) F0 values in Hz, 0 for unvoiced
    """
    # Convert mel_spec to numpy if it's a tensor
    if torch.is_tensor(mel_spec):
        mel_spec = mel_spec.detach().cpu().numpy()

    batch_size, n_mels, n_frames = mel_spec.shape
    f0_contour = np.zeros((batch_size, n_frames))

    # Create mel frequency bins
    mel_freqs = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels)
    hz_freqs = mel_to_hz(mel_freqs)

    for batch_idx in range(batch_size):
        for frame_idx in range(n_frames):
            frame = mel_spec[batch_idx, :, frame_idx]

            # Find peaks with minimum height threshold
            peaks, properties = find_peaks(
                frame, height=np.max(frame) * 0.1, distance=3
            )

            if len(peaks) > 0:
                # Get the strongest peak
                peak_heights = frame[peaks]
                strongest_peak_idx = peaks[np.argmax(peak_heights)]

                # Convert mel bin to Hz with interpolation for sub-bin accuracy
                if strongest_peak_idx > 0 and strongest_peak_idx < n_mels - 1:
                    # Parabolic interpolation around peak
                    y1, y2, y3 = frame[strongest_peak_idx - 1 : strongest_peak_idx + 2]
                    a = (y1 - 2 * y2 + y3) / 2
                    b = (y3 - y1) / 2
                    if a != 0:
                        peak_offset = -b / (2 * a)
                        refined_bin = strongest_peak_idx + peak_offset
                    else:
                        refined_bin = strongest_peak_idx
                else:
                    refined_bin = strongest_peak_idx

                # Interpolate frequency
                if refined_bin < n_mels - 1:
                    alpha = refined_bin - int(refined_bin)
                    f0_hz = (
                        hz_freqs[int(refined_bin)] * (1 - alpha)
                        + hz_freqs[int(refined_bin) + 1] * alpha
                    )
                else:
                    f0_hz = hz_freqs[int(refined_bin)]

                # Apply frequency range constraints
                if 80 <= f0_hz <= 2000:  # Typical F0 range for music/speech
                    f0_contour[batch_idx, frame_idx] = f0_hz

        # Post-processing: median filter to smooth F0 contour for each batch item
        f0_contour[batch_idx] = medfilt(f0_contour[batch_idx], kernel_size=5)

    return f0_contour


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


# Usage example:
# mel_spec = your_mel_spectrogram  # shape: (128, n_frames)
# f0 = extract_f0_from_mel(mel_spec)

if __name__ == "__main__":
    # Example usage
    from dataloader.insnotes_dataloader import get_dataloader

    train_loader = get_dataloader(
        # data_dir=os.path.join(self.data_dir, "train"),
        batch_size=8,
        num_workers=0,
        data_type=3,
        # shuffle=True,  # when distributed, shuffle is omitted
        # distributed=False,
    )
    for batch_data, batch_contents, batch_styles in train_loader:
        print(
            f"Batch data shape: {batch_data.shape}"
            f"Batch contents shape: {batch_contents.shape}"
            f"Batch styles shape: {batch_styles.shape}"
        )
        mel_spec = batch_data[0]  # Get the first sample's mel spectrogram
        f0_contour = extract_f0_from_mel(mel_spec[:, 0])
        a = 1
    # print(f"Extracted F0 contour: {f0_contour}")
    # print(f"F0 contour shape: {f0_contour.shape}")
