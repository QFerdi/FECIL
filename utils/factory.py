from learners.FECIL import FECIL


def get_model(model_name, args):
    name = model_name.lower()
    if(name == "fecil"):
        return FECIL(args)
    else:
        assert 0
