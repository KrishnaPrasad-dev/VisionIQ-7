from dataclasses import dataclass, field
from typing import List

@dataclass
class SystemConfig:
    mode: str = 'office'
    opening_time: str = '09:00'
    closing_time: str = '18:00'

    zones = [
{
"name": "Cash Counter",
"coords": [[150,120],[350,120],[350,280],[150,280]],
"threat_level": "high"
}
]

    mode_limits: dict = field(default_factory=lambda: {
        'home': 4,
        'shop': 10,
        'office': 20
    })

    dwell_time_threshold: int = 30

    prev_score: float = 0.0
    last_alert_time: float = 0.0
    alert_log: List[dict] = field(default_factory=list)
    activity_timeline: List[dict] = field(default_factory=list)

config = SystemConfig()
