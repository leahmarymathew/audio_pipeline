def normalize_metrics(snr: float, silence: float, dnsmos: float, whisper_conf: float, clipping: float):
    snr_norm = min(snr / 20.0, 1.0)
    speech_ratio = 1.0 - silence
    dnsmos_norm = dnsmos / 5.0
    whisper_norm = min((whisper_conf + 10.0) / 10.0, 1.0)
    unclipped = 1.0 - clipping
    return snr_norm, speech_ratio, dnsmos_norm, whisper_norm, unclipped


def composite_score(snr_norm: float, speech_ratio: float, dnsmos_norm: float, whisper_norm: float, unclipped: float):
    return (
        0.2 * snr_norm
        + 0.2 * speech_ratio
        + 0.3 * dnsmos_norm
        + 0.2 * whisper_norm
        + 0.1 * unclipped
    )
