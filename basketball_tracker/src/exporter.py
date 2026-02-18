"""
Export tracking results to various formats.
"""
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Optional

from .models import TrackingResult, VideoMetadata, Team
import config


class Exporter:
    """Exports tracking results to JSON and annotated video."""
    
    def __init__(self, output_dir: str = str(config.RESULTS_DIR)):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_json(
        self, 
        result: TrackingResult, 
        filename: str = "tracking_results.json"
    ) -> Path:
        """
        Export tracking results to JSON file.
        
        Args:
            result: TrackingResult object
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return output_path
    
    def create_video_writer(
        self,
        metadata: VideoMetadata,
        filename: str = "annotated_output.mp4",
        fps: Optional[float] = None
    ) -> cv2.VideoWriter:
        """
        Create a video writer for annotated output.
        
        Args:
            metadata: Video metadata for dimensions/fps
            filename: Output filename
            fps: Override FPS (None = use source fps)
        
        Returns:
            OpenCV VideoWriter object
        """
        output_path = self.output_dir / filename
        output_fps = fps if fps is not None else metadata.fps
        
        # Try multiple codecs for compatibility
        codecs = [
            config.OUTPUT_VIDEO_CODEC,  # mp4v (MPEG-4)
            "XVID",                      # Xvid
            "MJPG",                      # Motion JPEG
        ]
        
        writer = None
        working_codec = None
        
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    output_fps,
                    (metadata.width, metadata.height)
                )
                
                if writer.isOpened():
                    working_codec = codec
                    break
                else:
                    writer.release()
                    writer = None
            except Exception as e:
                print(f"Codec {codec} failed: {e}")
                continue
        
        if writer is None or not writer.isOpened():
            raise RuntimeError(
                f"Could not create video writer for {output_path}. "
                f"Tried codecs: {codecs}."
            )
        
        print(f"Using video codec: {working_codec}")
        return writer
    
    def save_frame(self, frame: np.ndarray, filename: str) -> Path:
        """Save a single frame as image."""
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), frame)
        return output_path
    
    def generate_summary(self, result: TrackingResult) -> dict:
        """Generate summary statistics from tracking results."""
        if not result.frames:
            return {"error": "No frames processed"}
        
        # Collect unique track IDs and team data
        all_track_ids = set()
        track_frame_counts = {}
        track_teams = {}
        team_counts = defaultdict(int)
        
        for frame in result.frames:
            for obj in frame.tracked_objects:
                all_track_ids.add(obj.track_id)
                track_frame_counts[obj.track_id] = track_frame_counts.get(obj.track_id, 0) + 1
                
                if obj.team != Team.UNKNOWN:
                    track_teams[obj.track_id] = obj.team.name
        
        # Count players per team
        for track_id, team_name in track_teams.items():
            team_counts[team_name] += 1
        
        detections_per_frame = [len(f.tracked_objects) for f in result.frames]
        
        summary = {
            "video_info": {
                "filepath": result.metadata.filepath,
                "duration_seconds": result.metadata.duration_seconds,
                "total_frames": result.metadata.total_frames,
                "fps": result.metadata.fps,
                "resolution": f"{result.metadata.width}x{result.metadata.height}"
            },
            "tracking_stats": {
                "frames_processed": len(result.frames),
                "unique_tracks": len(all_track_ids),
                "avg_detections_per_frame": sum(detections_per_frame) / len(detections_per_frame),
                "max_detections_per_frame": max(detections_per_frame),
                "min_detections_per_frame": min(detections_per_frame)
            },
            "team_stats": {
                "team_a_players": team_counts.get("TEAM_A", 0),
                "team_b_players": team_counts.get("TEAM_B", 0),
                "referees_detected": team_counts.get("REFEREE", 0),
                "unclassified": len(all_track_ids) - sum(team_counts.values())
            },
            "track_details": {
                track_id: {
                    "frames_visible": track_frame_counts[track_id],
                    "team": track_teams.get(track_id, "UNKNOWN")
                }
                for track_id in sorted(track_frame_counts.keys())
            }
        }
        
        return summary
    
    def export_summary(
        self, 
        result: TrackingResult, 
        filename: str = "tracking_summary.json"
    ) -> Path:
        """Export summary statistics to JSON."""
        summary = self.generate_summary(result)
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return output_path