"""CPU smoke test for models, losses, optimizers, metrics and saving (needs the full environment).

Skipped when TensorFlow is not installed, so the stdlib-only CI run stays light.
Run: python3 -m unittest tests.test_models_smoke -v
"""
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BATCH, CHANNELS, DIM = 2, 5, (64, 64)
MODELS = ["test_model", "unet", "unetlight", "srcnn_unet", "resunet", "resunetlight", "dlinknet", "transunet",
          "swin_unet"]
TRAJ_MODELS = ["unet_traj_type1", "unet_traj_type2", "resunet_traj_type1", "resunet_traj_type2",
               "dlinknet_traj_type1", "dlinknet_traj_type2", "transunet_traj_type1", "transunet_traj_type2",
               "swin_unet_traj_type1", "swin_unet_traj_type2"]
FUSIONS = ["average", "maximum", "multiply", "concat", "multiheadattention"]
LOSSES = ["binary_crossentropy", "dice_binary_crossentropy", "focal", "mse", "focal_custom", "contrastive"]


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow not installed")
class ModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        sys.path.insert(0, str(ROOT))
        cls.out = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def run_case(self, name, fusion="average", optimizer="SGD", loss="dice", save=False):
        import numpy as np
        import tensorflow as tf
        import tensorflow_addons as tfa
        from keras_unet_collection.activations import GELU
        from keras_unet_collection.transformer_layers import patch_extract, patch_embedding, \
            SwinTransformerBlock, patch_merging, patch_expanding
        from models.model_repository import ModelRepository
        from utils.custom_callbacks import TimeKeeper
        from utils.custom_metrics import MeanIoU, create_boundary

        tf.keras.backend.clear_session()
        srcnn = 2 if name == "srcnn_unet" else 0
        model = ModelRepository(name, DIM, CHANNELS, BATCH, srcnn_count=srcnn, optimizer=optimizer, l_rate=1e-4,
                                decay=1e-6, momentum=0.9, nesterov=True, loss=loss,
                                fusion_type=fusion).get_model()
        if srcnn:
            x = [np.random.rand(BATCH, *DIM, 1).astype("float32") for _ in range(srcnn)]
            x.append(np.random.rand(BATCH, *DIM, CHANNELS - srcnn).astype("float32"))
        else:
            x = np.random.rand(BATCH, *DIM, CHANNELS).astype("float32")
        # one-hot labels, as produced by RasterDataGenerator
        y = tf.keras.utils.to_categorical(np.random.randint(0, 2, (BATCH, *DIM)), num_classes=2).astype("float32")

        history = model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0, callbacks=[TimeKeeper(self.out + "/")])
        self.assertTrue(np.isfinite(history.history["loss"][0]))
        pred = model.predict(x, batch_size=BATCH, verbose=0)
        MeanIoU(batch_size=BATCH).update_state(tf.constant(y), tf.constant(pred))

        # per-sample metrics as in run_analysis.py
        p = (pred[0, :, :, 0] > 0.5).astype(int)
        t = y[0, :, :, 0].astype(int)
        for metric in [tf.keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.Accuracy(),
                       tf.keras.metrics.Precision(thresholds=0.5), tf.keras.metrics.Recall(thresholds=0.5),
                       tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='micro')]:
            metric.update_state(t, p)
            metric.result().numpy()
        tf.keras.metrics.MeanIoU(num_classes=2).update_state(create_boundary(t), create_boundary(p))

        if save:
            # both formats saved by run_model.py, reloaded as in the analysis scripts
            model.save(f"{self.out}/m_model")
            model.save(f"{self.out}/m_model.h5")
            custom_objects = {"GELU": GELU, "patch_extract": patch_extract, "patch_embedding": patch_embedding,
                              "SwinTransformerBlock": SwinTransformerBlock, "patch_merging": patch_merging,
                              "patch_expanding": patch_expanding}
            reloaded = tf.keras.models.load_model(f"{self.out}/m_model.h5", custom_objects=custom_objects,
                                                  compile=False)
            np.testing.assert_allclose(reloaded.predict(x, batch_size=BATCH, verbose=0), pred, atol=1e-5)

    def test_models(self):
        for name in MODELS:
            with self.subTest(model=name):
                self.run_case(name, save=True)

    def test_traj_models_fusions(self):
        for name in TRAJ_MODELS:
            for fusion in FUSIONS:
                with self.subTest(model=name, fusion=fusion):
                    self.run_case(name, fusion=fusion, optimizer="adam",
                                  save=fusion in ("average", "multiheadattention"))

    def test_losses_optimizers(self):
        for optimizer in ["SGD", "adam"]:
            for loss in LOSSES:
                with self.subTest(optimizer=optimizer, loss=loss):
                    self.run_case("unet", optimizer=optimizer, loss=loss, save=loss == "focal")


if __name__ == '__main__':
    unittest.main()
