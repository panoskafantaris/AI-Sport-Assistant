"""
Main pipeline – orchestrates all four phases per frame.

Court detection strategy (Option 3 + 4):
  - SceneDetector fires on every histogram/SSIM cut
  - AutoCourtDetector (Hough + colour mask) attempts auto-detection
    and scores confidence
  - confidence >= AUTO_THRESHOLD  → boundary stored automatically
  - confidence <  AUTO_THRESHOLD  → CourtCalibrator popup shown to user
  - User can confirm, skip (non-court scene), or quit
  - All boundaries stored in CalibrationMap with JSON persistence
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from .models.frame  import FrameData, TrackingResult
from .video.loader  import VideoLoader
from .court import (
    HomographyCalc, CourtCalibrator, CalibrationMode, CalibrationResult,
    SceneDetector, AutoCourtDetector, AUTO_THRESHOLD,
    CalibrationMap, SurfaceType,
)
from .players       import PlayerTracker
from .ball          import BallDetector, TrajectoryTracker, LandingPredictor
from .stats         import (MovementAnalyser, BallSpeedEstimator,
                            RallyDetector, KinesiologyAnalyser)
from .visualizer    import Visualizer
from .exporter      import Exporter
import config


class Pipeline:
    """Full tennis analysis pipeline with scene-aware court detection."""

    def __init__(
        self,
        output_dir:        str  = str(config.RESULTS_DIR),
        frame_skip:        int  = config.DEFAULT_SKIP,
        save_video:        bool = True,
        save_json:         bool = True,
        show_progress:     bool = True,
        doubles:           bool = False,
        enable_pose:       bool = False,
        interactive_court: bool = True,
        cal_map_path:      Optional[str] = None,
    ):
        self.output_dir        = Path(output_dir)
        self.frame_skip        = frame_skip
        self.save_video        = save_video
        self.save_json         = save_json
        self.show_progress     = show_progress
        self.doubles           = doubles
        self.interactive_court = interactive_court
        self.cal_map_path      = Path(cal_map_path) if cal_map_path else None

        # Court
        self._homography_calc  = HomographyCalc(doubles=doubles)
        self._calibrator       = CourtCalibrator(doubles=doubles)
        self._scene_detector   = SceneDetector()
        self._auto_detector    = AutoCourtDetector()
        self._cal_map          = CalibrationMap()

        # Other phases
        self._player_tracker  = PlayerTracker()
        self._ball_detector   = BallDetector()
        self._trajectory      = TrajectoryTracker()
        self._landing         = LandingPredictor()
        self._movement        = None   # fps-dependent, set in process()
        self._ball_speed      = None
        self._rally           = None
        self._kinesiology     = KinesiologyAnalyser(enabled=enable_pose)
        self._exporter        = Exporter(output_dir)

    # ── Public entry point ────────────────────────────────────────────────────

    def process(
        self,
        video_path:  str,
        max_frames:  Optional[int] = None,
        output_name: Optional[str] = None,
    ) -> TrackingResult:

        base = output_name or Path(video_path).stem
        cal_path = self.cal_map_path or self.output_dir / f"{base}_calibration.json"

        with VideoLoader(video_path) as loader:
            meta = loader.metadata
            fps  = meta.fps

            self._movement   = MovementAnalyser(fps=fps)
            self._ball_speed = BallSpeedEstimator(fps=fps)
            self._rally      = RallyDetector(fps=fps)

            # Load existing calibration map if available
            if cal_path.exists():
                self._cal_map.load(cal_path)
                print(f"[Pipeline] Loaded calibration map: {cal_path}")
            elif self.interactive_court:
                # Initial calibration on first frame
                first = loader.get_first_frame()
                if first is not None:
                    self._handle_scene_change(first, frame_number=0)
                    self._cal_map.save(cal_path)

            result = TrackingResult(metadata=meta)

            writer = None
            if self.save_video:
                writer = self._exporter.create_video_writer(
                    meta, filename=f"{base}_annotated.mp4"
                )

            frames_iter = loader.frames(skip=self.frame_skip, max_frames=max_frames)
            if self.show_progress:
                frames_iter = tqdm(frames_iter, total=meta.total_frames,
                                   desc="Processing", unit="frames")

            quit_requested = False
            for fn, ts, frame in frames_iter:
                # ── Scene change check ────────────────────────────────────────
                event = self._scene_detector.update(frame, fn, ts)
                if event is not None and self.interactive_court:
                    print(f"\n[Scene] Change at frame {fn}  "
                          f"({event.trigger}, diff={event.hist_diff:.2f})")
                    quit_requested = self._handle_scene_change(frame, fn)
                    if quit_requested:
                        break
                    self._cal_map.save(cal_path)

                frame_data = self._process_frame(frame, fn, ts)
                result.frames.append(frame_data)

                if writer:
                    writer.write(self._annotate(frame, frame_data))

            if writer:
                writer.release()

        result.rallies = self._rally.completed_rallies

        if self.save_json:
            self._exporter.export_json(result, filename=f"{base}_tracking.json")

        print(f"\n{self._cal_map.summary()}")
        return result

    # ── Scene change handler ──────────────────────────────────────────────────

    def _handle_scene_change(self, frame, frame_number: int) -> bool:
        """
        Try auto-detection. If confidence is low, open the calibration UI.
        Returns True if the user requested a full quit.
        """
        boundary, confidence, surface = self._auto_detector.detect(frame)

        surface_name = surface.surface.value if surface else "unknown"
        print(f"[Court]  Auto-detection: surface={surface_name}  "
              f"conf={confidence:.2f}  threshold={AUTO_THRESHOLD:.2f}")

        if boundary is not None and confidence >= AUTO_THRESHOLD:
            # Auto-accepted
            homogr = self._homography_calc.compute(boundary)
            self._cal_map.add_scene(
                start_frame   = frame_number,
                boundary      = boundary,
                surface       = surface.surface if surface else SurfaceType.UNKNOWN,
                confidence    = confidence,
                auto_detected = True,
            )
            print(f"[Court]  Auto-accepted (conf={confidence:.2f})")
            return False

        if not self.interactive_court:
            # Non-interactive: store whatever we got (may be None)
            self._cal_map.add_scene(
                start_frame   = frame_number,
                boundary      = boundary,
                surface       = surface.surface if surface else SurfaceType.UNKNOWN,
                confidence    = confidence,
                auto_detected = True,
                is_court      = boundary is not None,
            )
            return False

        # ── Confidence too low → show calibration UI ──────────────────────────
        print(f"[Court]  Confidence too low → opening calibration UI")
        cal_result, new_boundary, new_homogr = self._calibrator.run(
            frame,
            mode      = CalibrationMode.SCENE,
            auto_conf = confidence,
            surface   = surface_name,
        )

        if cal_result == CalibrationResult.QUIT:
            return True     # signal quit to main loop

        is_court = cal_result == CalibrationResult.CONFIRMED
        self._cal_map.add_scene(
            start_frame   = frame_number,
            boundary      = new_boundary,
            surface       = surface.surface if surface else SurfaceType.UNKNOWN,
            confidence    = 1.0 if is_court else 0.0,
            auto_detected = False,
            is_court      = is_court,
        )
        return False

    # ── Per-frame processing ──────────────────────────────────────────────────

    def _process_frame(self, frame, fn: int, ts: float) -> FrameData:
        fd = FrameData(frame_number=fn, timestamp_ms=ts)

        # Phase 1: look up boundary for this frame from the calibration map
        fd.court = self._cal_map.get_boundary(fn)
        if fd.court is not None:
            homogr = self._homography_calc.compute(fd.court)
        else:
            homogr = None
        fd.homography = homogr

        # Attach surface type as an extra
        fd.extras["surface"] = self._cal_map.get_surface(fn).value

        # Phase 2: players
        players = self._player_tracker.track(frame, court=fd.court)
        players = self._movement.update(players, homography=homogr)
        players = self._kinesiology.analyse(frame, players)
        fd.players = players

        # Phase 3: ball
        ball = self._ball_detector.detect(frame, frame_number=fn, timestamp_ms=ts)
        if ball and homogr and homogr.is_ready():
            try:
                ball.world_x, ball.world_y = homogr.image_to_world(ball.x, ball.y)
            except Exception:
                pass
        fd.ball = ball

        traj = self._trajectory.update(ball)
        fd.trajectory = traj
        fd.landing = self._landing.predict(traj, fd.court, homogr)

        # Phase 4: stats
        speed = self._ball_speed.update(ball, homogr)
        fd.extras["ball_speed_ms"] = round(speed, 2)
        self._rally.update(fn, ball, speed)
        fd.extras["rally_count"] = len(self._rally.completed_rallies)

        return fd

    # ── Annotation ────────────────────────────────────────────────────────────

    def _annotate(self, frame, fd: FrameData):
        out = frame.copy()
        Visualizer.draw_court(out, fd.court)
        Visualizer.draw_players(out, fd.players)
        Visualizer.draw_ball(out, fd.ball)
        Visualizer.draw_landing(out, fd.landing)
        surface = fd.extras.get("surface", "")
        Visualizer.draw_info(
            out, fd.frame_number,
            fd.extras.get("ball_speed_ms", 0.0),
            fd.extras.get("rally_count", 0),
            surface=surface,
        )
        return out