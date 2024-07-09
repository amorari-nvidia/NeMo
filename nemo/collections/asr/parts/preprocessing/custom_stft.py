import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class CustomBatchedSTFT(nn.Module):
    def __init__(self, n_fft, hop_length=None, win_length=None, window=None,
                 center=True, mode='reflect', normalized=False, onesided=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.mode = mode
        self.normalized = normalized
        self.onesided = onesided

    def forward(self, x):
        magnitude, phase = self.stft(x, self.n_fft, self.hop_length, self.win_length, self.window,
                                     self.center, self.mode, self.normalized, self.onesided)
        return magnitude, phase

    def stft(self, input, n_fft, hop_length=None, win_length=None, window=None,
             center=True, mode='reflect', normalized=False, onesided=True):

        if window is None:
            window = torch.ones(n_fft, device=input.device)
            print("A window was not provided. A rectangular window will be applied,"
                  " which is known to cause spectral leakage. Other windows such as torch.hann_window"
                  " or torch.hamming_window are recommended to reduce spectral leakage."
                  " To suppress this warning and use a rectangular window, explicitly set"
                  " `window=torch.ones(n_fft, device=<device>)`.")

        if not window.device == input.device:
            window.to(input.device)
            #raise ValueError(f"stft input and window must be on the same device but got input on {input.device} and window on {window.device}")

        hop_length = hop_length if hop_length is not None else n_fft // 4
        win_length = win_length if win_length is not None else n_fft

        if not input.is_floating_point() and not input.is_complex():
            raise ValueError(f"stft expected a tensor of floating point or complex values, but got {input.dtype}")

        if input.dim() > 2 or input.dim() < 1:
            raise ValueError(f"stft expected a 1D or 2D tensor, but got {input.dim()}D tensor")

        if input.dim() == 1:
            input = input.unsqueeze(0)

        if center:
            pad_amount = n_fft // 2
            input = F.pad(input, (pad_amount, pad_amount), mode=mode)

        batch = input.size(0)
        length = input.size(1)

        if n_fft <= 0 or n_fft > length:
            raise ValueError(f"stft expected 0 < n_fft <= {length}, but got n_fft={n_fft}")

        if hop_length <= 0:
            raise ValueError(f"stft expected hop_length > 0, but got hop_length={hop_length}")

        if win_length <= 0 or win_length > n_fft:
            raise ValueError(f"stft expected 0 < win_length <= n_fft, but got win_length={win_length}")

        if window.dim() != 1 or window.size(0) != win_length:
            raise ValueError(f"stft expected a 1D window tensor of size equal to win_length={win_length}, but got window with size {window.size()}")

        if win_length < n_fft:
            left = (n_fft - win_length) // 2
            window_ = torch.zeros(n_fft, device=input.device)
            window_[left:left + win_length] = window
        else:
            window_ = window

        num_frames = 1 + (length - n_fft) // hop_length
        input = input.as_strided((batch, num_frames, n_fft), (input.stride(0), hop_length * input.stride(1), input.stride(1)))

        if window is not None:
            input = input * window_

        if onesided:
            fft_func = torch.fft.rfft if not input.is_complex() else torch.fft.fft
        else:
            fft_func = torch.fft.fft

        output = fft_func(input, n=n_fft, dim=-1, norm='ortho' if normalized else 'backward')
        output = output.transpose(1, 2)

        magnitude = torch.abs(output)
        phase = torch.angle(output)

        return magnitude, phase


def compare_stft(x, n_fft, hop_length, win_length, window, reps):
    
    custom_stft = CustomBatchedSTFT(n_fft, hop_length, win_length, window)

    custom_magnitude, custom_phase = custom_stft(x)
    start_time = time.time()
    for _ in range(reps):
        custom_stft(x)
    end_time = time.time()
    time_custom = end_time - start_time
        
    torch_output = torch.stft(x, n_fft, hop_length, win_length, window, return_complex=True)
    start_time = time.time()
    for _ in range(reps):
        torch.stft(x, n_fft, hop_length, win_length, window, return_complex=True)
    end_time = time.time()
    time_torch = end_time - start_time
    
    torch_magnitude = torch.abs(torch_output)
    torch_phase = torch.angle(torch_output)

    # Print the first 100 elements of both tensors
    #print(f"custom_magnitude: {custom_magnitude.flatten()[:100]}")
    #print(f"torch_magnitude: {torch_magnitude.flatten()[:100]}")

    #print(f"custom_phase: {custom_phase.flatten()[:100]}")
    #print(f"torch_phase: {torch_phase.flatten()[:100]}")
    
    # Ensure the outputs have the same shape
    min_frames = min(custom_magnitude.shape[-1], torch_magnitude.shape[-1])
    custom_magnitude = custom_magnitude[..., :min_frames]
    custom_phase = custom_phase[..., :min_frames]
    torch_magnitude = torch_magnitude[..., :min_frames]
    torch_phase = torch_phase[..., :min_frames]
    
    max_diff_mag = torch.max(torch.abs(custom_magnitude - torch_magnitude))
    mean_diff_mag = torch.mean(torch.abs(custom_magnitude - torch_magnitude))
    
    max_diff_phase = torch.max(torch.abs(custom_phase - torch_phase))
    mean_diff_phase = torch.mean(torch.abs(custom_phase - torch_phase))
    
    return max_diff_mag.item(), mean_diff_mag.item(), max_diff_phase.item(), mean_diff_phase.item(), time_custom, time_torch

def generate_test_signal(length, sample_rate):
    t = torch.linspace(0, length, int(length * sample_rate))
    signal = torch.sin(2 * np.pi * 440 * t) + 0.5 * torch.sin(2 * np.pi * 880 * t)
    return signal

def run_tests():
    print("Running STFT comparison tests...")
    
    test_cases = [
        # Standard test cases
        (1, 16000, 512, 256, 512, 'hann'),
        (1, 22050, 1024, 512, 1024, 'hamming'),
        (2, 44100, 2048, 1024, 2048, 'blackman'),
        (4, 8000, 256, 128, 256, 'hann'),
        (16, 8000, 256, 128, 256, 'hamming'),
        (32, 8000, 256, 128, 256, 'bartlett'),
        # Edge cases
        #(1, 16000, 512, 256, 256, 'hann'),    # win_length < n_fft
        #(1, 16000, 512, 512, 1024, 'hann'),   # win_length > n_fft (invalid case)
        #(1, 16000, 512, 256, 128, 'hann'),    # win_length < n_fft and hop_length > win_length
        #(1, 16000, 512, 0, 512, 'hann'),      # hop_length = 0 (invalid case)
        #(1, 16000, 0, 256, 512, 'hann'),      # n_fft = 0 (invalid case)
        #(1, 16000, 512, 256, 512, 'none'),    # Invalid window type
    ]
    
    for batch_size, sample_rate, n_fft, hop_length, win_length, window_type in test_cases:
        
        print(f"\nTest case: batch_size={batch_size}, sample_rate={sample_rate}, n_fft={n_fft}, "
              f"hop_length={hop_length}, win_length={win_length}, window_type={window_type}")
        try:
            x = torch.stack([generate_test_signal(5, sample_rate) for _ in range(batch_size)])
            window = getattr(torch, f'{window_type}_window')(win_length)
            max_diff_mag, mean_diff_mag, max_diff_phase, mean_diff_phase, time_custom, time_torch = compare_stft(x, n_fft, hop_length, win_length, window, reps=10)
            
            print(f"Magnitude difference: Max {max_diff_mag:.6f}  Mean {mean_diff_mag:.6f}")
            print(f"Phase difference: Max {max_diff_phase:.6f}  Mean {mean_diff_phase:.6f}")
            print(f"Time custom: {time_custom:0.6f} Time torch: {time_torch:.6f} Time custom/torch: {time_custom/time_torch:.6f}")
            
        except Exception as e:
            print(f"Exception occurred: {e}")

if __name__ == "__main__":
    run_tests()
