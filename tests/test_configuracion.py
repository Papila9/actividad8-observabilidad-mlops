import unittest

from src.configuracion import load_config


class ConfigurationTests(unittest.TestCase):
    def test_loads_versioned_configs(self):
        self.assertIn("cooldown_seconds", load_config("drift_alerts"))
        self.assertIn("acceptance", load_config("model_lifecycle"))

    def test_rejects_unknown_config(self):
        with self.assertRaises(ValueError):
            load_config("slo")


if __name__ == "__main__":
    unittest.main()
