"""
Part 2：信道均衡实验

学生需要实现迫零均衡 (ZF) 和最小均方误差均衡 (MMSE)。
"""

import numpy as np

from utils import (
    generate_qpsk_symbols,
    transmit_through_channel,
    add_awgn,
    calculate_ber,
    plot_eye_diagram,
    plot_mse_vs_snr,
)

# 多径信道参数（学生无需修改）
CHANNEL = np.array([0.8, 0.3, 0.1])
CARRIER_FREQ = 100
SYMBOL_RATE = 10
SAMPLES_PER_SYMBOL = 8


def zero_forcing_equalizer(received, channel, num_taps):
    """迫零均衡器实现。"""
    conv_matrix = np.zeros((len(received) + num_taps - 1, num_taps), dtype=complex)
    for i in range(num_taps):
        conv_matrix[i:i + len(channel), i] = channel

    target = np.zeros(conv_matrix.shape[0])
    target[len(channel) // 2 + num_taps // 2] = 1

    conv_slice = conv_matrix[:len(target), :]
    zf_weights = np.linalg.pinv(conv_slice) @ target

    equalized = np.convolve(received, zf_weights, mode='same')
    return equalized, zf_weights


def mmse_equalizer(received, channel, num_taps, snr_db):
    """MMSE 均衡器实现。"""
    snr_linear = 10 ** (snr_db / 10)
    conv_matrix = np.zeros((len(received) + num_taps - 1, num_taps), dtype=complex)
    for i in range(num_taps):
        conv_matrix[i:i + len(channel), i] = channel

    target = np.zeros(conv_matrix.shape[0])
    target[len(channel) // 2 + num_taps // 2] = 1
    conv_slice = conv_matrix[:len(target), :]

    cov_matrix = conv_slice.T @ conv_slice + (1 / snr_linear) * np.eye(num_taps)
    mmse_weights = np.linalg.inv(cov_matrix) @ conv_slice.T @ target
    equalized = np.convolve(received, mmse_weights, mode='same')
    return equalized, mmse_weights


def run_equalization_demo():
    """运行 Part 2 演示并生成结果图。"""
    print('=' * 60)
    print('Part 2：信道均衡实验')
    print('=' * 60)

    num_symbols = 1000
    num_taps_equalizer = 5

    try:
        symbols = generate_qpsk_symbols(num_symbols)
        baseband, _ = transmit_through_channel(
            symbols, CHANNEL, CARRIER_FREQ, SYMBOL_RATE, SAMPLES_PER_SYMBOL
        )

        snr_eye = 10
        noisy_baseband = add_awgn(baseband, snr_db=snr_eye)
        plot_eye_diagram(
            noisy_baseband, SAMPLES_PER_SYMBOL,
            title=f'接收信号眼图 (SNR={snr_eye}dB)',
            save_path='equalization_rx_eye.png'
        )

        # 只保留 MMSE，删除无用变量
        eq_mmse, _ = mmse_equalizer(noisy_baseband, CHANNEL, num_taps_equalizer, snr_eye)

        plot_eye_diagram(
            eq_mmse, SAMPLES_PER_SYMBOL,
            title=f'MMSE 均衡后眼图 (SNR={snr_eye}dB)',
            save_path='equalization_equalized_eye.png'
        )

        snr_range = np.arange(0, 21, 2)
        mse_list = []
        ber_list = []

        for snr_db in snr_range:
            noisy = add_awgn(baseband, snr_db)
            eq, _ = mmse_equalizer(noisy, CHANNEL, num_taps_equalizer, snr_db)
            symbols_est = eq[::SAMPLES_PER_SYMBOL]
            symbols_est = symbols_est[:len(symbols)]
            mse = np.mean(np.abs(symbols_est - symbols) ** 2)
            mse_list.append(mse)
            ber = calculate_ber(symbols, symbols_est)
            ber_list.append(ber)

        plot_mse_vs_snr(snr_range, mse_list, ber_list, 'equalization_mse_ber.png')
        print('✅ 均衡实验图像已生成')

    except Exception as error:
        print(f'❌ Part 2 运行失败：{error}')
        