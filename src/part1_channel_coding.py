"""
Part 1：信道编码实验

学生需要完成 Hamming(7,4) 编码、伴随式计算和单比特纠错译码。
选做内容包括卷积码编码和 Viterbi 硬判决译码。
"""

import numpy as np

from utils import (
    binary_symmetric_channel,
    calculate_ber,
    generate_bits,
    plot_ber_curve,
)

HAMMING_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=int)

HAMMING_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=int)


def hamming74_encode(bits):
    """Hamming(7,4) 系统码编码。"""
    bits = np.asarray(bits, dtype=int)
    if bits.ndim != 1:
        raise ValueError('bits 必须是一维数组')
    if len(bits) % 4 != 0:
        raise ValueError('Hamming(7,4) 要求输入长度为 4 的倍数')
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    blocks = bits.reshape(-1, 4)
    encoded_blocks = blocks @ HAMMING_G % 2
    return encoded_blocks.flatten()


def hamming74_syndrome(codewords):
    """计算 Hamming(7,4) 码字的伴随式。"""
    codewords = np.asarray(codewords, dtype=int)
    if codewords.ndim == 1:
        if len(codewords) % 7 != 0:
            raise ValueError('码字长度必须是 7 的倍数')
        codewords = codewords.reshape(-1, 7)
    if codewords.shape[1] != 7:
        raise ValueError('每个 Hamming(7,4) 码字长度必须为 7')

    syndromes = codewords @ HAMMING_H.T % 2
    return syndromes


def hamming74_decode(received):
    """Hamming(7,4) 单比特纠错译码。"""
    received = np.asarray(received, dtype=int)
    if received.ndim != 1 or len(received) % 7 != 0:
        raise ValueError('received 必须是一维数组，长度为 7 的倍数')

    codewords = received.reshape(-1, 7)
    syndromes = hamming74_syndrome(codewords)

    for i in range(codewords.shape[0]):
        s = syndromes[i]
        if not np.all(s == 0):
            for col_idx in range(7):
                if np.array_equal(s, HAMMING_H[:, col_idx]):
                    codewords[i, col_idx] ^= 1
                    break

    decoded_bits = codewords[:, :4].flatten()
    return decoded_bits


def convolutional_encode(bits):
    """(2,1,3) 卷积码编码，生成多项式 g1=111, g2=101。"""
    bits = np.asarray(bits, dtype=int)
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    padded_bits = np.concatenate([bits, np.array([0, 0], dtype=int)])
    shift_reg = np.array([0, 0], dtype=int)
    encoded = []

    for bit in padded_bits:
        v1 = bit ^ shift_reg[0] ^ shift_reg[1]
        v2 = bit ^ shift_reg[1]

        encoded.append(v1)
        encoded.append(v2)

        shift_reg = np.array([bit, shift_reg[0]], dtype=int)

    return np.array(encoded, dtype=int)


def viterbi_decode_hard(received_bits):
    """(2,1,3) 卷积码 Viterbi 硬判决译码。"""
    received_bits = np.asarray(received_bits, dtype=int)
    if len(received_bits) % 2 != 0:
        raise ValueError('卷积码接收序列长度必须是 2 的倍数')

    received = received_bits.reshape(-1, 2)
    n_steps = len(received)
    n_states = 4

    path_metrics = np.full(n_states, np.inf)
    path_metrics[0] = 0
    back_pointers = np.zeros((n_steps, n_states), dtype=int)

    for t in range(n_steps):
        current_rx = received[t]
        new_metrics = np.full(n_states, np.inf)

        for current_state in range(n_states):
            if path_metrics[current_state] == np.inf:
                continue

            for input_bit in [0, 1]:
                next_state = ((current_state << 1) | input_bit) & 0b11
                s1 = (current_state >> 1) & 1
                s0 = current_state & 1

                expected_v1 = input_bit ^ s1 ^ s0
                expected_v2 = input_bit ^ s0
                expected = np.array([expected_v1, expected_v2])

                hamming_dist = np.sum(current_rx != expected)
                total_metric = path_metrics[current_state] + hamming_dist

                if total_metric < new_metrics[next_state]:
                    new_metrics[next_state] = total_metric
                    back_pointers[t, next_state] = current_state

        path_metrics = new_metrics

    decoded = []
    current_state = 0
    for t in range(n_steps - 1, -1, -1):
        prev_state = back_pointers[t, current_state]
        input_bit = (current_state >> 1) & 1
        decoded.append(input_bit)
        current_state = prev_state

    decoded = np.array(decoded[::-1], dtype=int)
    decoded = decoded[:-2]

    return decoded


def run_coding_demo():
    """运行 Part 1 演示并生成 BER 曲线。"""
    print('=' * 60)
    print('Part 1：信道编码实验')
    print('=' * 60)

    error_probabilities = np.array([0.001, 0.003, 0.01, 0.03, 0.06, 0.1])
    uncoded_ber = []
    coded_ber = []

    try:
        bits = generate_bits(4000, seed=2026)
        bits = bits[: len(bits) // 4 * 4]
        encoded = hamming74_encode(bits)

        for idx, prob in enumerate(error_probabilities):
            uncoded_rx = binary_symmetric_channel(bits, prob, seed=100 + idx)
            encoded_rx = binary_symmetric_channel(encoded, prob, seed=200 + idx)
            decoded = hamming74_decode(encoded_rx)
            uncoded_ber.append(calculate_ber(bits, uncoded_rx))
            coded_ber.append(calculate_ber(bits, decoded))

        plot_ber_curve(
            error_probabilities,
            {'未编码': uncoded_ber, 'Hamming(7,4)': coded_ber},
            'Hamming(7,4) 编码前后 BER 对比',
            'coding_ber_curve.png',
        )
        print('✅ 已生成 results/coding_ber_curve.png')
    except NotImplementedError as error:
        print(f'⏸️ 尚未完成核心函数：{error}')
    except Exception as error:
        print(f'❌ Part 1 运行失败：{error}')
        