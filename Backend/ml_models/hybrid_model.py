from ml_models.informer.informer import informer_test
from ml_models.gru.gru import gru_test

def test():
    informer = informer_test()
    gru = gru_test()
    return [{"informer": informer}, {"gru": gru}]
