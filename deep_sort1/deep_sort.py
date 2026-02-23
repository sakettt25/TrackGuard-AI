"""
DeepSort wrapper to use deep-sort-realtime package
"""
from deep_sort_realtime.deepsort_tracker import DeepSort as DeepSortRealtime
import numpy as np


class Track:
    """Wrapper for track object"""
    def __init__(self, track_id, ltwh):
        self.track_id = track_id
        self.ltwh = ltwh
        self.hits = 1
        
    def to_tlwh(self):
        """Convert to top-left-width-height format"""
        return self.ltwh


class Tracker:
    """Wrapper for tracker object"""
    def __init__(self):
        self.tracks = []


class DeepSort:
    """DeepSort wrapper class"""
    def __init__(self, model_path=None, max_age=30, n_init=3, nms_max_overlap=1.0, 
                 max_cosine_distance=0.2, nn_budget=None, override_track_class=None):
        # Initialize deep-sort-realtime tracker
        self._deepsort = DeepSortRealtime(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            override_track_class=override_track_class,
            embedder="mobilenet",  # Use mobilenet embedder
            half=True,
            bgr=True,
        )
        self.tracker = Tracker()  # Separate wrapper for track results
        
    def update(self, bbox_xywh, confidences, image):
        """
        Update tracker with detections
        
        Args:
            bbox_xywh: List of bounding boxes in [x, y, w, h] format
            confidences: List of confidence scores
            image: Current frame
            
        Returns:
            List of tracks
        """
        # Convert to format expected by deep-sort-realtime
        # It expects ([left, top, width, height], confidence, class_id)
        detections = []
        for bbox, conf in zip(bbox_xywh, confidences):
            if len(bbox) == 4:
                x, y, w, h = bbox
                # deep-sort-realtime expects [left, top, width, height]
                detections.append(([x, y, w, h], conf, 0))  # class_id=0 for person
        
        # Update tracker (uses the real DeepSortRealtime instance)
        tracks = self._deepsort.update_tracks(detections, frame=image)
        
        # Convert tracks to expected format and store in our wrapper
        self.tracker.tracks = []
        result_tracks = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            raw_id = track.track_id
            # Convert string track_id to integer for compatibility
            track_id = hash(raw_id) % (10**6) if isinstance(raw_id, str) else int(raw_id)
            ltwh = track.to_ltwh()
            
            # Create Track object
            track_obj = Track(track_id, ltwh)
            self.tracker.tracks.append(track_obj)
            result_tracks.append(track_obj)
        
        return result_tracks
