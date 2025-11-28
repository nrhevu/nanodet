# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import onnx
import onnxsim
import torch

from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

try:
    import onnxruntime as ort
    import numpy as np
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


def generate_ouput_names(head_cfg):
    cls_names, dis_names = [], []
    for stride in head_cfg.strides:
        cls_names.append("cls_pred_stride_{}".format(stride))
        dis_names.append("dis_pred_stride_{}".format(stride))
    return cls_names + dis_names


def verify_onnx_export(model, onnx_path, dummy_input, logger):
    """Verify that ONNX export matches PyTorch model output."""
    if not ONNXRUNTIME_AVAILABLE:
        logger.log("Skipping verification (onnxruntime not installed)")
        return

    logger.log("Verifying ONNX export...")

    # Get PyTorch output
    model.eval()
    with torch.no_grad():
        pytorch_output = model(dummy_input)
        if isinstance(pytorch_output, (list, tuple)):
            pytorch_output = pytorch_output[0]
        pytorch_output = pytorch_output.cpu().numpy()

    # Get ONNX output
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_array = dummy_input.detach().cpu().numpy()
    onnx_output = ort_session.run(None, {"data": input_array})[0]

    import pdb; pdb.set_trace()

    # Compare outputs
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()

    logger.log(f"Max difference: {max_diff:.6f}")
    logger.log(f"Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-3:
        logger.log("✓ Verification passed!")
    else:
        logger.log(f"⚠ Verification warning: difference {max_diff:.6f} exceeds 1e-3")


def main(config, model_path, output_path, input_shape=(320, 320)):
    logger = Logger(-1, config.save_dir, False)
    model = build_model(config.model)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_model_weight(model, checkpoint, logger)
    if config.model.arch.backbone.name == "RepVGG":
        deploy_config = config.model
        deploy_config.arch.backbone.update({"deploy": True})
        deploy_model = build_model(deploy_config)
        from nanodet.model.backbone.repvgg import repvgg_det_model_convert

        model = repvgg_det_model_convert(model, deploy_model)

    dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, input_shape[0], input_shape[1])
    )

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=18,
        input_names=["data"],
        output_names=["output"],
    )
    logger.log("finished exporting onnx ")

    logger.log("start simplifying onnx ")
    input_data = {"data": dummy_input.detach().cpu().numpy()}
    model_sim, flag = onnxsim.simplify(output_path, input_data=input_data)
    if flag:
        onnx.save(model_sim, output_path)
        logger.log("simplify onnx successfully")
    else:
        logger.log("simplify onnx failed")

    # Verify the export
    verify_onnx_export(model, output_path, dummy_input, logger)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth or .ckpt model to onnx.",
    )
    parser.add_argument("--cfg_path", type=str, help="Path to .yml config file.")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to .ckpt model."
    )
    parser.add_argument(
        "--out_path", type=str, default="nanodet.onnx", help="Onnx model output path."
    )
    parser.add_argument(
        "--input_shape", type=str, default=None, help="Model intput shape."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.cfg_path
    model_path = args.model_path
    out_path = args.out_path
    input_shape = args.input_shape
    load_config(cfg, cfg_path)
    if input_shape is None:
        input_shape = cfg.data.train.input_size
    else:
        input_shape = tuple(map(int, input_shape.split(",")))
        assert len(input_shape) == 2
    if model_path is None:
        model_path = os.path.join(cfg.save_dir, "model_best/model_best.ckpt")
    main(cfg, model_path, out_path, input_shape)
    print("Model saved to:", out_path)
