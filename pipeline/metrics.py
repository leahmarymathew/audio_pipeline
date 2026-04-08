from pipeline.dnsmos import compute_dnsmos

def compute_metrics(audio, sr, asr_conf, lang_pred, lang_conf, expected_lang):

    import numpy as np

    # -------- WADA-SNR --------
    def compute_wada_snr(audio):
        audio = audio + 1e-10
        abs_audio = np.abs(audio)

        mean_abs = np.mean(abs_audio)
        log_abs = np.log(abs_audio)

        mean_log = np.mean(log_abs)

        snr = 20 * (np.log10(mean_abs) - mean_log)
        return snr

    # -------- C50 (Clarity Index) --------
    def compute_c50(audio, sr):
        energy = audio ** 2
        split = int(0.05 * sr)  # 50ms

        early = np.sum(energy[:split]) + 1e-9
        late = np.sum(energy[split:]) + 1e-9

        c50 = 10 * np.log10(early / late)
        return c50

    # -------- Basic Metrics --------
    duration = len(audio) / sr
    clipping = np.sum(np.abs(audio) > 0.99) / len(audio)

    # -------- Silence --------
    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)

    energy = []
    for j in range(0, len(audio) - frame_length, hop_length):
        frame = audio[j:j + frame_length]
        energy.append(np.mean(frame**2))

    energy = np.array(energy)
    silence = 1 if len(energy) == 0 else np.sum(energy < 0.0005) / len(energy)

    # -------- WADA-SNR --------
    snr = compute_wada_snr(audio)

    # -------- C50 --------
    c50 = compute_c50(audio, sr)

    # -------- DNSMOS --------
    sig, bak, ovr = compute_dnsmos(audio, sr)
    dnsmos = ovr

    # -------- Normalization --------
    snr_norm = min(max(snr / 20, 0), 1)
    speech_ratio = 1 - silence
    dnsmos_norm = min(max(dnsmos / 5, 0), 1)

    # -------- Score (UPDATED with ASR) --------
    score = (
        0.35 * dnsmos_norm +
        0.2 * speech_ratio +
        0.2 * snr_norm +
        0.15 * (1 - clipping) +
        0.1 * asr_conf
    )

    # -------- Decision Logic --------
    decision = "KEEP"
    reason = []

    # 🔴 HARD FILTERS

    # DNSMOS
    if dnsmos < 2.5:
        decision = "REJECT"
        reason.append(f"low_dnsmos({dnsmos:.2f})")
    lang_map = {
        "en": "english",
        "hi": "hindi",
        "ta": "tamil",
        "bn": "bengali",
        "mr": "marathi",
        "gu": "gujarati"
    }
    # Language mismatch
    mapped_lang = lang_map.get(lang_pred, "unknown")

    if mapped_lang != expected_lang:
        score -= 0.1   # penalty instead of reject
        reason.append(f"lang_mismatch({lang_pred})")


    # Low ASR confidence
    if asr_conf < 0.2:
        decision = "REJECT"
        reason.append(f"low_asr_conf({asr_conf:.2f})")

    # Silence
    if silence > 0.5:
        decision = "REJECT"
        reason.append(f"too_much_silence({silence:.2f})")

    # 🔴 SOFT FILTER (only if not already rejected)
    if decision != "REJECT":
        if score < 0.55:
            decision = "REJECT"
            reason.append(f"low_score({score:.2f})")

    return {
        "duration": duration,
        "snr": snr,
        "c50": c50,
        "silence": silence,
        "clipping": clipping,
        "dnsmos": dnsmos,
        "dnsmos_sig": sig,
        "dnsmos_bak": bak,
        "speech_ratio": speech_ratio,
        "asr_conf": asr_conf,          # 🔥 NEW
        "lang_pred": lang_pred,        # 🔥 NEW
        "lang_conf": lang_conf,        # 🔥 NEW
        "score": score,
        "decision": decision,
        "reason": ",".join(reason) if reason else ""
    }