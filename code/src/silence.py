import logging
import warnings


def silence():
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        category=UserWarning,
        module="sklearn.utils.validation",
    )

    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    warnings.filterwarnings("ignore", message=".*pin_memory.*")
    warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated", category=UserWarning)

    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


def silence_darts_warnings():
    logging.getLogger("darts").setLevel(logging.ERROR)
