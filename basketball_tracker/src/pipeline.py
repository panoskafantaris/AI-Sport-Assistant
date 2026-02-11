"""
Main pipeline that orchestrates video processing.
"""
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from .models import FrameData, TrackingResult, Team
from .video_loader import VideoLoader
from .tracker import Tracker
from .visualizer import Visualizer
from .exporter import Exporter
from .team_classifier import TeamClassifier
import config


class Pipeline:
    """Orchestrates the full video processing pipeline."""
    
    def __init__(
        self,
        output_dir: str = str(config.RESULTS_DIR),
        frame_skip: int = config.DEFAULT_FRAME_SKIP,
        save_video: bool = True,
        save_json: bool = True,
        show_progress: bool = True,
        enable_team_classification: bool = True
    ):
        """
        Initialize pipeline.
        
        Args:
            output_dir: Directory for output files
            frame_skip: Frames to skip between processing
            save_video: Whether to save annotated video
            save_json: Whether to save JSON results
            show_progress: Whether to show progress bar
            enable_team_classification: Whether to classify players into teams
        """
        self.output_dir = Path(output_dir)
        self.frame_skip = frame_skip
        self.save_video = save_video
        self.save_json = save_json
        self.show_progress = show_progress
        self.enable_team_classification = enable_team_classification
        
        # Initialize components
        self.tracker = Tracker()
        self.visualizer = Visualizer()
        self.exporter = Exporter(str(self.output_dir))
        
        # Team classifier (Phase 2)
        self.team_classifier = TeamClassifier() if enable_team_classification else None
    
    def process(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        output_name: Optional[str] = None
    ) -> TrackingResult:
        """
        Process a video file end-to-end.
        
        Args:
            video_path: Path to input video
            max_frames: Maximum frames to process (None = all)
            output_name: Base name for output files (None = use input name)
        
        Returns:
            TrackingResult with all tracking data
        """
        # Setup
        video_loader = VideoLoader(video_path)
        metadata = video_loader.metadata
        
        if output_name is None:
            output_name = Path(video_path).stem
        
        # Initialize result
        result = TrackingResult(metadata=metadata, frames=[])
        
        # Setup video writer if needed
        video_writer = None
        if self.save_video:
            video_writer = self.exporter.create_video_writer(
                metadata,
                filename=f"{output_name}_annotated.mp4"
            )
        
        # Calculate total frames for progress bar
        total = metadata.total_frames // (self.frame_skip + 1)
        if max_frames:
            total = min(total, max_frames)
        
        # Process frames
        self.tracker.reset()
        if self.team_classifier:
            self.team_classifier.reset()
        frames_written = 0
        
        with video_loader:
            frame_iterator = video_loader.frames(
                skip=self.frame_skip,
                max_frames=max_frames
            )
            
            if self.show_progress:
                frame_iterator = tqdm(
                    frame_iterator,
                    total=total,
                    desc="Processing frames"
                )
            
            for frame_num, timestamp_ms, frame in frame_iterator:
                # Track objects
                tracked_objects = self.tracker.track(frame)
                
                # Classify teams (Phase 2)
                if self.team_classifier:
                    tracked_objects = self.team_classifier.classify_frame(
                        frame, tracked_objects
                    )
                    # Update visualizer with team colors if calibrated
                    if self.team_classifier.is_calibrated:
                        self.visualizer.set_team_colors(
                            self.team_classifier.team_colors
                        )
                
                # Store frame data
                frame_data = FrameData(
                    frame_number=frame_num,
                    timestamp_ms=timestamp_ms,
                    tracked_objects=tracked_objects
                )
                result.frames.append(frame_data)
                
                # Visualize and write frame
                if video_writer:
                    annotated = self.visualizer.draw_tracks(frame, tracked_objects)
                    
                    # Get team counts for display
                    team_counts = None
                    if self.team_classifier and self.team_classifier.is_calibrated:
                        team_counts = self.team_classifier.get_team_stats(tracked_objects)
                    
                    annotated = self.visualizer.draw_frame_info(
                        annotated,
                        frame_num,
                        len(tracked_objects),
                        team_counts=team_counts
                    )
                    video_writer.write(annotated)
                    frames_written += 1
        
        # Cleanup video writer
        if video_writer:
            video_writer.release()
            print(f"Saved annotated video: {self.output_dir}/{output_name}_annotated.mp4")
            print(f"Frames written: {frames_written}")
        
        # Export JSON results
        if self.save_json:
            json_path = self.exporter.export_json(
                result,
                filename=f"{output_name}_tracking.json"
            )
            print(f"Saved tracking data: {json_path}")
            
            summary_path = self.exporter.export_summary(
                result,
                filename=f"{output_name}_summary.json"
            )
            print(f"Saved summary: {summary_path}")
        
        return result
    
    def process_frame(self, frame) -> FrameData:
        """
        Process a single frame (for testing or real-time use).
        
        Args:
            frame: BGR image array
        
        Returns:
            FrameData with tracking results
        """
        tracked_objects = self.tracker.track(frame)
        
        # Classify teams if enabled
        if self.team_classifier:
            tracked_objects = self.team_classifier.classify_frame(
                frame, tracked_objects
            )
        
        return FrameData(
            frame_number=0,
            timestamp_ms=0.0,
            tracked_objects=tracked_objects
        )