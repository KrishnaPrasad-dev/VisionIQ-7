import time

class DwellTracker:
    def __init__(self, threshold_seconds=30):
        self.threshold = threshold_seconds
        self.entry_times = {}
        self.loitering = {}

    def update(self, track_ids, zone_name, in_zone_flags):
        alerts = []
        now = time.time()

        for track_id, in_zone in zip(track_ids, in_zone_flags):
            key = (int(track_id), zone_name)

            if in_zone:
                if key not in self.entry_times:
                    self.entry_times[key] = now

                dwell = now - self.entry_times[key]

                if dwell >= self.threshold:
                    alerts.append((int(track_id), zone_name, round(dwell)))
                    self.loitering[int(track_id)] = zone_name
            else:
                if key in self.entry_times:
                    del self.entry_times[key]
                if int(track_id) in self.loitering:
                    del self.loitering[int(track_id)]

        return alerts

    def get_dwell_time(self, track_id, zone_name):
        key = (int(track_id), zone_name)
        if key in self.entry_times:
            return round(time.time() - self.entry_times[key])
        return 0

    def is_loitering(self, track_id):
        return int(track_id) in self.loitering

    def clear_lost_tracks(self, active_ids):
        active = set(int(i) for i in active_ids)
        to_delete = [k for k in self.entry_times if k[0] not in active]
        for k in to_delete:
            del self.entry_times[k]
        self.loitering = {
            k: v for k, v in self.loitering.items() if k in active
        }

dwell_tracker = DwellTracker()