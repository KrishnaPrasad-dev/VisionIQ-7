import time
from datetime import datetime
from config import config

ZONE_THREAT_SCORES = {
    'low':      20,
    'medium':   35,
    'high':     50,
    'critical': 65
}

_critical_since  = None
_last_status     = 'SAFE'
_status_hold_until = 0          # don't downgrade status before this time
STATUS_HOLD_SECS = 3.0          # hold a status for at least 3 seconds
CRITICAL_HOLD_SECONDS = 2


def is_after_hours():
    return True   # DEMO MODE


def is_weekend_office():
    return True


def calculate_threat_score(
    person_count,
    zone_results,
    after_hours,
    loitering_count=0,
    motion=None
):
    motion = motion or {}
    score  = 0.0

    score += person_count * 22

    zone_hits = 0
    for zone_name, triggered, threat_level in zone_results:
        if triggered:
            score += ZONE_THREAT_SCORES.get(threat_level, 30)
            zone_hits += 1

    if loitering_count > 0:
        score += loitering_count * 15

    if motion.get('running'):   score += 20
    if motion.get('panic'):     score += 25
    if motion.get('abandoned'): score += 18

    multiplier = 1.0
    if zone_hits > 0:
        multiplier *= 2.0
    elif person_count > 0:
        multiplier *= 1.6

    score *= multiplier
    return min(score, 100)


# ── SMOOTHING ────────────────────────────────────────────────────────────────
# Uses a rolling buffer of last N raw scores and exponential smoothing.
# This kills frame-to-frame jumping completely.

_score_buffer = []
BUFFER_SIZE   = 6   # average over last 6 frames (~0.5s at 12fps)

def apply_score_decay(new_score):
    global _score_buffer

    _score_buffer.append(new_score)
    if len(_score_buffer) > BUFFER_SIZE:
        _score_buffer.pop(0)

    # Weighted average — recent frames count more
    weights = list(range(1, len(_score_buffer) + 1))
    buffered = sum(s * w for s, w in zip(_score_buffer, weights)) / sum(weights)

    # Then apply exponential smoothing on top
    if buffered > config.prev_score:
        # Rising — moderate speed
        smoothed = 0.75 * buffered + 0.25 * config.prev_score
    else:
        # Falling — very slow, holds alert state longer
        smoothed = 0.15 * buffered + 0.85 * config.prev_score

    config.prev_score = smoothed
    return smoothed


# ── STATUS with hold-time ─────────────────────────────────────────────────────
def get_status(score):
    global _last_status, _status_hold_until

    now = time.time()

    if score >= 55:
        new_status = 'CRITICAL'
    elif score >= 25:
        new_status = 'SUSPICIOUS'
    else:
        new_status = 'SAFE'

    # Only allow downgrade after hold time expires
    severity = {'SAFE': 0, 'SUSPICIOUS': 1, 'CRITICAL': 2}
    if severity[new_status] < severity[_last_status]:
        if now < _status_hold_until:
            # Too soon — keep current status
            return _last_status

    # Status is changing up or hold expired — accept it
    if new_status != _last_status:
        _status_hold_until = now + STATUS_HOLD_SECS

    _last_status = new_status
    return new_status


def update_critical_timer(status):
    global _critical_since
    if status == 'CRITICAL':
        if _critical_since is None:
            _critical_since = time.time()
    else:
        _critical_since = None


def critical_sustained():
    if _critical_since is None:
        return False
    return (time.time() - _critical_since) >= CRITICAL_HOLD_SECONDS


def check_cooldown():
    return (time.time() - config.last_alert_time) > 8


def log_alert(score, status, person_count, snapshot_path=None, extra=None):
    entry = {
        'timestamp':     datetime.now().isoformat(),
        'status':        status,
        'score':         round(score),
        'person_count':  person_count,
        'snapshot_path': snapshot_path,
        'extra':         extra or {}
    }
    config.alert_log.insert(0, entry)
    config.alert_log = config.alert_log[:10]
    config.activity_timeline.insert(0, {
        'time':   datetime.now().strftime('%H:%M:%S'),
        'status': status
    })
    config.activity_timeline = config.activity_timeline[:20]