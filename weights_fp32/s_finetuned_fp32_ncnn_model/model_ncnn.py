import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.rand(1, 3, 640, 640, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
        net.load_param("/content/coffee_day3_focus/day3_s_focus_v1/quantization/20261119_160817_80645d/export_work/ncnn_fp32/best_ncnn_model/model.ncnn.param")
        net.load_model("/content/coffee_day3_focus/day3_s_focus_v1/quantization/20261119_160817_80645d/export_work/ncnn_fp32/best_ncnn_model/model.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))
            _, out1 = ex.extract("out1")
            out.append(torch.from_numpy(np.array(out1)).unsqueeze(0))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":
    print(test_inference())
