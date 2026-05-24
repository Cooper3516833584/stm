import time

from FlightController.Components.LDRadar_Driver import LD_Radar


def test_new_radar_not_fresh():
    radar = LD_Radar(name="test")
    assert radar.get_last_frame_age_s() is None
    assert not radar.is_fresh(max_age_s=0.5)


def test_manual_last_frame_age():
    radar = LD_Radar(name="test")
    now = time.perf_counter()
    with radar._latency_lock:
        radar._last_valid_frame_host_time_s = now - 0.1
    assert radar.is_fresh(max_age_s=0.5, now_s=now)
    assert not radar.is_fresh(max_age_s=0.05, now_s=now)

