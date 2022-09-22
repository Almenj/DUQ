""" Archie Luxton, aol21, https://github.com/ese-msc-2021/irp-aol21, 2022 """
import sgld
import mc_dropout
import post
import pre
from flask import Flask, request
import torch
import numpy as np
import sys

sys.path.append('./')
try:
    from duq import pre, post, mc_dropout, sgld  # noqa
except ImportError or ModuleNotFoundError:
    NotImplemented
    #    import pre, post, mc_dropout, sgld

app = Flask(__name__)

MC = torch.load("../trained_models/for_prediction/MC_Dropout")
SG = torch.load("../trained_models/for_prediction/SGLD")
reference = torch.load("../trained_models/for_prediction/MCDropout_reference")


@app.route('/predict', methods=['POST'])
def hello():
    res = request.get_json(force=True)
    vec = [
        res["numx"],
        res["numy"],
        res["numz"],
        res["dimx"],
        res["dimy"],
        res["dimz"]]
    method = res["method"]
    if method == "MC":
        model = MC
    elif method == "SGLD":
        model = SG
    elif method == "reference":
        model = reference
    vec_arr = np.array(vec)
    s, m, sd = model.make_prediction(vec_arr)
    res = {"mean": m.item(), "sd": sd.item()}
    # print(f'\n## MEAN = {res["mean"]} ## \n')
    # print(f'\n## MEAN = {res["sd"]} ## \n')
    return res


if __name__ == '__main__':
    app.run()
