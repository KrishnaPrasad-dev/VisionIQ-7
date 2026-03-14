import cv2
import numpy as np
import supervision as sv
from config import config

STATUS_COLORS = {
    'SAFE':       (0, 200, 0),
    'SUSPICIOUS': (0, 200, 255),
    'CRITICAL':   (0, 0, 255)
}

ZONE_COLORS = {
    'low':      (0, 255, 0),
    'medium':   (0, 165, 255),
    'high':     (0, 0, 255),
    'critical': (128, 0, 128)
}

box_annotator   = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5)

def annotate_frame(frame, detections, zone_results, score, status,
                   loitering_ids=None, motion=None):
    annotated     = frame.copy()
    color         = STATUS_COLORS[status]
    loitering_ids = loitering_ids or []
    motion        = motion or {}

    # ── DRAW ZONES ────────────────────────────────────────────
    for zone_name, triggered, threat_level in zone_results:
        zone_data = next((z for z in config.zones if z['name'] == zone_name), None)
        if not zone_data:
            continue

        pts        = np.array(zone_data['coords'], np.int32)
        zone_color = ZONE_COLORS.get(threat_level, (0, 0, 255))

        cv2.polylines(annotated, [pts], True, zone_color, 2)

        if triggered:
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], zone_color)
            cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)

        cv2.putText(annotated, f'{zone_name} [{threat_level.upper()}]',
            (pts[0][0], pts[0][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2)

    # ── DRAW OPTICAL FLOW VECTORS ─────────────────────────────
    if motion.get('flow_vectors') and (motion.get('running') or motion.get('panic')):
        vec_color = (0, 0, 255) if motion.get('panic') else (0, 165, 255)
        for prev_pt, curr_pt in motion['flow_vectors'][:30]:  # max 30 vectors
            p1 = (int(prev_pt[0]), int(prev_pt[1]))
            p2 = (int(curr_pt[0]), int(curr_pt[1]))
            cv2.arrowedLine(annotated, p1, p2, vec_color, 1, tipLength=0.4)

    # ── DRAW BOUNDING BOXES ───────────────────────────────────
    if len(detections) > 0:
        labels = []
        for i, conf in enumerate(detections.confidence):
            track_id     = int(detections.tracker_id[i]) if detections.tracker_id is not None else i
            is_loitering = track_id in loitering_ids
            label        = f'#{track_id} {"LOITERING" if is_loitering else f"{conf:.2f}"}'
            labels.append(label)

        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

    # ── HUD BAR ───────────────────────────────────────────────
    h, w      = annotated.shape[:2]
    hud_lines = []

    if motion.get('running'):   hud_lines.append('RUNNING')
    if motion.get('panic'):     hud_lines.append('PANIC')
    if motion.get('abandoned'): hud_lines.append('ABANDONED OBJECT')
    if loitering_ids:           hud_lines.append(f'LOITERING:{len(loitering_ids)}')

    hud_height = 55 + (20 if hud_lines else 0)
    cv2.rectangle(annotated, (0, 0), (w, hud_height), (20, 20, 20), -1)

    cv2.putText(annotated, f'THREAT: {round(score)} | {status}',
        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(annotated, f'People: {len(detections)} | Mode: {config.mode.upper()}',
        (w - 310, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)

    if hud_lines:
        cv2.putText(annotated, ' | '.join(hud_lines),
            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    return annotated