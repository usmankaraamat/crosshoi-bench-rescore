"""Download the released HOI-specialist predictions on HICO-DET into ./predictions."""
import os, shutil
from huggingface_hub import hf_hub_download

FILES = ["ADA_CM-test_default_L.json", "CMMP-test_default_L.json", "LAIN-test_default_B.json",
         "HOLa-test_default_L.json", "CMD_SE-test_default_B.json"]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions")
os.makedirs(out, exist_ok=True)
for f in FILES:
    path = hf_hub_download("chelsielei/CrossHOI_Bench", f"HOI_pred_hicodet/{f}")
    shutil.copy(path, os.path.join(out, f))
    print("ok", f)
