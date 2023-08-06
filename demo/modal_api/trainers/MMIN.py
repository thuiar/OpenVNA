from modal_api.configs.MMIN import Options
from modal_api.models import create_model

def eval(model, feature):
    model.eval()
    model.set_input(feature)
    model.test()
    pred = round(float(model.pred.detach().cpu().numpy()[0][0]), 4)
    return pred

def robust_test(feature):
    seed = 1111
    opt = Options().parse(1111)  
    test_model_path = f"/home/sharing/disk1/zhangbaozheng/encoder/emnlp/mmin/{seed}"
    opt.isTrain = False
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.load_networks(test_model_path)
    tst_result = eval(model, feature)

    
    return tst_result
