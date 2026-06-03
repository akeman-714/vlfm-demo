import os

p = "vlfm/utils/vlfm_trainer.py"
s = open(p, encoding="utf-8").read()

if "VLFM_DUMP_RGB_DIR" in s:
    print("already injected; skipping")
    raise SystemExit(0)

anchor = '            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)'
assert s.count(anchor) == 1, ("anchor count", s.count(anchor))

block = (
    '            if "VLFM_DUMP_RGB_DIR" in os.environ:\n'
    '                import imageio, numpy as _np\n'
    '                _ddir = os.environ["VLFM_DUMP_RGB_DIR"]\n'
    '                os.makedirs(_ddir, exist_ok=True)\n'
    '                _gstep = getattr(self, "_vlfm_dump_step", 0)\n'
    '                for _i in range(len(observations)):\n'
    '                    _rgb = observations[_i]["rgb"]\n'
    '                    if hasattr(_rgb, "cpu"):\n'
    '                        _rgb = _rgb.cpu().numpy()\n'
    '                    _rgb = _np.asarray(_rgb).astype("uint8")\n'
    '                    _act = step_data[_i] if _i < len(step_data) else -1\n'
    '                    imageio.imwrite(os.path.join(_ddir, "env%d_step%04d_act%s.png" % (_i, _gstep, _act)), _rgb)\n'
    '                _arr = _np.asarray(observations[0]["rgb"]).astype("float32")\n'
    '                print("[RGBDUMP] step %d shape %s mean %.3f std %.3f" % (_gstep, str(_arr.shape), _arr.mean(), _arr.std()), flush=True)\n'
    '                self._vlfm_dump_step = _gstep + 1\n'
)

s = s.replace(anchor, block + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
print("injected OK")
