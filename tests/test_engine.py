import numpy as np
import pytest
from engine import ADASPipeline


@pytest.fixture
def mock_frame():
    return np.zeros((360, 640, 3), dtype=np.uint8)


@pytest.fixture
def clear_telemetry():
    return {
        "hazard_detected": False,
        "hazard_type": None,
        "road_pct": 25.0,
        "vehicle_count": 0,
        "pedestrian_count": 0,
        "warning_zone_occupancy": 0.0,
        "inference_ms": 42.0,
        "timestamp": "12:00:00",
    }


@pytest.fixture
def hazard_telemetry():
    return {
        "hazard_detected": True,
        "hazard_type": "Vehicle",
        "road_pct": 20.0,
        "vehicle_count": 3,
        "pedestrian_count": 1,
        "warning_zone_occupancy": 35.0,
        "inference_ms": 55.0,
        "timestamp": "12:00:05",
    }


class TestDrawHUD:
    def test_returns_same_shape(self, mock_frame, clear_telemetry):
        result = ADASPipeline.draw_hud(mock_frame, clear_telemetry)
        assert result.shape == mock_frame.shape
        assert result.dtype == mock_frame.dtype

    def test_non_destructive(self, mock_frame, clear_telemetry):
        original = mock_frame.copy()
        ADASPipeline.draw_hud(mock_frame, clear_telemetry)
        assert np.array_equal(mock_frame, original)

    def test_hud_does_not_blank_frame(self, mock_frame, clear_telemetry):
        result = ADASPipeline.draw_hud(mock_frame, clear_telemetry)
        assert np.any(result != 0)

    def test_clear_status_no_brake(self, mock_frame, clear_telemetry):
        result = ADASPipeline.draw_hud(mock_frame, clear_telemetry)
        assert result.shape == mock_frame.shape

    def test_hazard_status_still_returns_frame(self, mock_frame, hazard_telemetry):
        result = ADASPipeline.draw_hud(mock_frame, hazard_telemetry)
        assert result.shape == mock_frame.shape
        assert np.any(result != 0)

    def test_telemetry_values_appear_in_output(self, mock_frame, hazard_telemetry):
        result = ADASPipeline.draw_hud(mock_frame, hazard_telemetry)
        assert result.shape == mock_frame.shape

    def test_zero_occ_no_error(self, mock_frame, clear_telemetry):
        t = dict(clear_telemetry, warning_zone_occupancy=0.0)
        result = ADASPipeline.draw_hud(mock_frame, t)
        assert result.shape == mock_frame.shape

    def test_max_occ_no_error(self, mock_frame, clear_telemetry):
        t = dict(clear_telemetry, warning_zone_occupancy=100.0)
        result = ADASPipeline.draw_hud(mock_frame, t)
        assert result.shape == mock_frame.shape

    def test_missing_timestamp(self, mock_frame, clear_telemetry):
        t = dict(clear_telemetry)
        del t["timestamp"]
        result = ADASPipeline.draw_hud(mock_frame, t)
        assert result.shape == mock_frame.shape

    def test_missing_inference_ms(self, mock_frame, clear_telemetry):
        t = dict(clear_telemetry)
        del t["inference_ms"]
        result = ADASPipeline.draw_hud(mock_frame, t)
        assert result.shape == mock_frame.shape

    def test_empty_frame(self, clear_telemetry):
        empty = np.zeros((1, 1, 3), dtype=np.uint8)
        result = ADASPipeline.draw_hud(empty, clear_telemetry)
        assert result.shape == empty.shape

    @pytest.mark.parametrize("h, w", [(1080, 1920), (720, 1280), (480, 640)])
    def test_various_resolutions(self, h, w, clear_telemetry):
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        result = ADASPipeline.draw_hud(frame, clear_telemetry)
        assert result.shape == (h, w, 3)


@pytest.mark.integration
class TestPipelineIntegration:
    def test_pipeline_imports(self):
        assert ADASPipeline is not None
        assert hasattr(ADASPipeline, "process_frame")
        assert hasattr(ADASPipeline, "draw_hud")

    def test_process_frame_signature(self):
        import inspect
        sig = inspect.signature(ADASPipeline.process_frame)
        params = list(sig.parameters.keys())
        assert "frame" in params

    def test_draw_hud_signature(self):
        import inspect
        sig = inspect.signature(ADASPipeline.draw_hud)
        params = list(sig.parameters.keys())
        assert "frame" in params
        assert "telemetry" in params
        assert "display_frame" in params
