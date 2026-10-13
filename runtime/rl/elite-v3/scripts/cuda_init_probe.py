import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-4d9f95ae-dfba-0ba5-9aad-09372fa19208"
t = time.time()
import torch
print("import", round(time.time() - t, 1), flush=True)
print(torch.cuda.device_count(), torch.cuda.get_device_name(0), round(time.time() - t, 1), flush=True)
x = torch.ones(1024, 1024, device="cuda:0"); print("alloc ok", float(x.sum()), round(time.time() - t, 1), flush=True)
