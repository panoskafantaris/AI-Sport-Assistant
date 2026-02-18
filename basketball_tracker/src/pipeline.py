"""
Main pipeline with dynamic court detection.

1. Click court floor → set court color (one time)
2. Click players → assign teams (3 each)

Court boundaries are detected automatically each frame based on color.
"""
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from .models import FrameData, TrackingResult, Team, TrackedObject
from .video_loader import VideoLoader
from .tracker import Tracker
from .visualizer import Visualizer
from .exporter import Exporter
from .team_classifier import TeamClassifier, InteractiveCalibrator, CourtDetector
import config


class Pipeline:
    """Video processing pipeline with dynamic court detection."""
    
    def __init__(
        self,
        output_dir: str = str(config.RESULTS_DIR),
        frame_skip: int = config.DEFAULT_FRAME_SKIP,
        save_video: bool = True,
        save_json: bool = True,
        show_progress: bool = True,
        enable_team_classification: bool = True,
        interactive_calibration: bool = False
    ):
        """Initialize pipeline."""
        self.output_dir = Path(output_dir)
        self.frame_skip = frame_skip
        self.save_video = save_video
        self.save_json = save_json
        self.show_progress = show_progress
        self.enable_team_classification = enable_team_classification
        self.interactive_calibration = interactive_calibration
        
        # Components
        self.tracker = Tracker()
        self.visualizer = Visualizer()
        self.exporter = Exporter(str(self.output_dir))
        
        # Calibration
        self.team_classifier: Optional[TeamClassifier] = None
        self.court_detector: Optional[CourtDetector] = None
        self.calibrator: Optional[InteractiveCalibrator] = None
        
        if interactive_calibration:
            self.calibrator = InteractiveCalibrator()
    
    def _filter_by_court(
        self,
        frame,
        tracked_objects: List[TrackedObject]
    ) -> List[TrackedObject]:
        """Filter to only players on court (dynamic per frame)."""
        if not self.court_detector or not self.court_detector.is_color_set:
            return tracked_objects
        
        # Update court mask for this frame
        self.court_detector.update_frame(frame)
        
        filtered = []
        for obj in tracked_objects:
            bbox = obj.bbox.to_int_tuple()
            if self.court_detector.is_on_court(frame, bbox):
                filtered.append(obj)
        return filtered
    
    def _run_calibration(self, video_path: str) -> bool:
        """
        Run two-phase interactive calibration.
        
        Phase 1: Court calibration (full video, can restart if needed)
        Phase 2: Team assignment (restart video)
        """
        print("\n" + "="*60)
        print("INTERACTIVE CALIBRATION - TWO PHASES")
        print("="*60)
        print("")
        print("PHASE 1: COURT CALIBRATION")
        print("        LEFT-click:  Add COURT color (2pt, 3pt, paint, etc.)")
        print("        RIGHT-click: Add OUT-OF-BOUNDS color (stands, ads, etc.)")
        print("        → Click as many times as needed")
        print("        → Video will play through, or press ENTER to finish early")
        print("        → Review results, restart if not satisfied")
        print("")
        print("PHASE 2: TEAM ASSIGNMENT (after court is set)")
        print("        → Video restarts from beginning")
        print("        → Click players, press A/B/R to assign teams")
        print("")
        print("Press Q at any time to quit")
        print("="*60 + "\n")
        
        result = None
        
        # Phase 1: Court calibration (can restart multiple times)
        while True:
            print("Starting court calibration pass...")
            self.tracker.reset()
            cal_loader = VideoLoader(video_path)
            
            with cal_loader:
                frame_count = 0
                total_frames = cal_loader.metadata.total_frames
                
                for frame_num, timestamp_ms, frame in cal_loader.frames():
                    frame_count += 1
                    is_last = (frame_count >= total_frames - 1)
                    
                    tracked_objects = self.tracker.track(frame)
                    result = self.calibrator.update(frame, tracked_objects, is_last_frame=is_last)
                    
                    if result == "quit":
                        print("Calibration cancelled")
                        self.calibrator.cleanup()
                        return False
                    
                    elif result == "restart":
                        break  # Break inner loop to restart
                    
                    elif result == "phase2":
                        break  # Break inner loop to go to phase 2
            
            # Decide what to do after video loop
            if result == "restart":
                print("Restarting video for more court samples...\n")
                continue  # Restart outer while loop
            elif result == "phase2":
                break  # Exit outer loop, proceed to phase 2
            else:
                # Video ended without explicit action, go to review
                pass
        
        # Phase 2: Team assignment
        print("\n" + "-"*40)
        print("PHASE 2: TEAM ASSIGNMENT")
        print("Court calibration complete. Restarting video...")
        print("-"*40 + "\n")
        
        self.tracker.reset()
        team_loader = VideoLoader(video_path)
        
        with team_loader:
            for frame_num, timestamp_ms, frame in team_loader.frames():
                tracked_objects = self.tracker.track(frame)
                result = self.calibrator.update(frame, tracked_objects)
                
                if result == "quit":
                    print("Calibration cancelled")
                    self.calibrator.cleanup()
                    return False
                
                elif result == "complete":
                    self._finalize_calibration()
                    return True
        
        # Video ended during team assignment
        self.calibrator.cleanup()
        
        if self.calibrator.is_complete():
            self._finalize_calibration()
            return True
        
        print("Calibration incomplete - not all teams assigned")
        return False
    
    def _finalize_calibration(self) -> None:
        """Save and set up components after successful calibration."""
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE")
        print("="*60)
        
        # Print court info
        court_colors = self.calibrator.court_detector.get_court_colors()
        oob_colors = self.calibrator.court_detector.get_oob_colors()
        print(f"\nCourt colors ({len(court_colors)} samples):")
        for i, rgb in enumerate(court_colors):
            print(f"  {i+1}. RGB{rgb}")
        
        if oob_colors:
            print(f"\nOut-of-bounds colors ({len(oob_colors)} samples):")
            for i, rgb in enumerate(oob_colors):
                print(f"  {i+1}. RGB{rgb}")
        
        # Print team info
        print(f"\n{self.calibrator.color_store}")
        print("="*60 + "\n")
        
        # Save
        self.calibrator.color_store.save()
        self.calibrator.court_detector.save()
        
        # Set up components
        self.team_classifier = TeamClassifier(
            color_store=self.calibrator.get_color_store()
        )
        self.court_detector = self.calibrator.get_court_detector()
    
    def process(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        output_name: Optional[str] = None
    ) -> TrackingResult:
        """Process video with dynamic court detection."""
        video_loader = VideoLoader(video_path)
        metadata = video_loader.metadata
        
        if output_name is None:
            output_name = Path(video_path).stem
        
        # Interactive calibration
        if self.interactive_calibration and self.enable_team_classification:
            self.tracker.reset()
            if not self._run_calibration(video_path):
                print("Continuing without team classification...")
                self.enable_team_classification = False
            # Recreate video_loader after calibration
            video_loader = VideoLoader(video_path)
        elif self.enable_team_classification:
            # Try loading saved
            self.team_classifier = TeamClassifier()
            self.court_detector = CourtDetector()
            
            if self.team_classifier.load_calibration():
                print("Loaded team calibration")
                print(self.team_classifier.color_store)
            else:
                print("No team calibration found. Use --interactive")
                self.enable_team_classification = False
            
            if self.court_detector.load():
                colors = self.court_detector.get_all_colors()
                print(f"Loaded court ({len(colors)} color samples)")
            else:
                print("No court color found")
                self.court_detector = None
        
        # Initialize result
        result = TrackingResult(metadata=metadata, frames=[])
        
        # Video writer
        video_writer = None
        if self.save_video:
            video_writer = self.exporter.create_video_writer(
                metadata, filename=f"{output_name}_annotated.mp4"
            )
        
        # Update visualizer colors
        if self.team_classifier and self.team_classifier.is_calibrated:
            self.visualizer.set_team_colors(self.team_classifier.team_colors)
        
        # Progress
        total = metadata.total_frames // (self.frame_skip + 1)
        if max_frames:
            total = min(total, max_frames)
        
        self.tracker.reset()
        
        print(f"\nProcessing: {video_path}")
        
        with video_loader:
            frame_iter = video_loader.frames(
                skip=self.frame_skip,
                max_frames=max_frames
            )
            
            if self.show_progress:
                frame_iter = tqdm(frame_iter, total=total, desc="Processing")
            
            for frame_num, timestamp_ms, frame in frame_iter:
                # Track
                tracked_objects = self.tracker.track(frame)
                
                # Filter by court (dynamic per frame)
                tracked_objects = self._filter_by_court(frame, tracked_objects)
                
                # Classify teams
                if self.enable_team_classification and self.team_classifier:
                    tracked_objects = self.team_classifier.classify_frame(
                        frame, tracked_objects
                    )
                
                # Store
                frame_data = FrameData(
                    frame_number=frame_num,
                    timestamp_ms=timestamp_ms,
                    tracked_objects=tracked_objects
                )
                result.frames.append(frame_data)
                
                # Write video
                if video_writer:
                    annotated = frame.copy()
                    
                    # Draw dynamic court boundary (yellow)
                    if self.court_detector and self.court_detector.is_color_set:
                        annotated = self.court_detector.draw_boundary(
                            annotated, (0, 255, 255), 2
                        )
                    
                    # Draw tracks
                    annotated = self.visualizer.draw_tracks(annotated, tracked_objects)
                    
                    # Draw info
                    team_counts = None
                    if self.team_classifier:
                        team_counts = self.team_classifier.get_team_stats(tracked_objects)
                    
                    annotated = self.visualizer.draw_frame_info(
                        annotated, frame_num, len(tracked_objects),
                        team_counts=team_counts
                    )
                    
                    video_writer.write(annotated)
        
        # Cleanup
        if video_writer:
            video_writer.release()
            print(f"Saved: {self.output_dir}/{output_name}_annotated.mp4")
        
        if self.save_json:
            self.exporter.export_json(result, f"{output_name}_tracking.json")
            self.exporter.export_summary(result, f"{output_name}_summary.json")
        
        return result