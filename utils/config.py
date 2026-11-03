import configparser
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DataCfg:
    dataset: str
    data_root: str
    img_height: int
    img_width: int
    channels: str
    val_split: float
    num_workers: int
    augment: bool

@dataclass
class ModelCfg:
    patch_size: int
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    depth: int
    mlp_ratio: float
    dropout: float
    num_classes: int
    use_class_token: bool
    pos_embed: str

@dataclass
class TrainCfg:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    betas: tuple
    warmup_epochs: int
    lr_scheduler: str
    seed: int
    amp: bool
    device: str
    log_interval: int

@dataclass
class SaveCfg:
    out_dir: str
    save_weights: bool
    save_best: bool
    # Mirrors spicecimx/config.py SaveCfg (used only by inference).
    clean_workspace: bool = True

@dataclass
class Cfg:
    data: DataCfg
    model: ModelCfg
    train: TrainCfg
    save: SaveCfg

def _get_bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "y", "t")

def load_cfg(path: str) -> Cfg:
    # Same parser settings as spicecimx.config.make_parser: strip inline
    # '#'/';' comments, treat '%' literally.
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"), interpolation=None)
    with open(path, "r", encoding="utf-8-sig") as f:     # tolerate a BOM
        cp.read_file(f)
    betas = tuple(float(x) for x in cp.get("TRAIN", "betas").split(","))
    data = DataCfg(
        dataset=cp.get("DATA", "dataset"),
        data_root=cp.get("DATA", "data_root"),
        img_height=cp.getint("DATA", "img_height"),
        img_width=cp.getint("DATA", "img_width"),
        channels=cp.get("DATA", "channels"),
        val_split=cp.getfloat("DATA", "val_split"),
        num_workers=cp.getint("DATA", "num_workers"),
        augment=_get_bool(cp.get("DATA", "augment"))
    )
    model = ModelCfg(
        patch_size=cp.getint("MODEL", "patch_size"),
        d_model=cp.getint("MODEL", "d_model"),
        n_heads=cp.getint("MODEL", "n_heads"),
        d_k=cp.getint("MODEL", "d_k"),
        d_v=cp.getint("MODEL", "d_v"),
        depth=cp.getint("MODEL", "depth"),
        mlp_ratio=cp.getfloat("MODEL", "mlp_ratio"),
        dropout=cp.getfloat("MODEL", "dropout"),
        num_classes=cp.getint("MODEL", "num_classes"),
        use_class_token=_get_bool(cp.get("MODEL", "use_class_token")),
        pos_embed=cp.get("MODEL", "pos_embed"),
    )
    train = TrainCfg(
        epochs=cp.getint("TRAIN", "epochs"),
        batch_size=cp.getint("TRAIN", "batch_size"),
        lr=cp.getfloat("TRAIN", "lr"),
        weight_decay=cp.getfloat("TRAIN", "weight_decay"),
        betas=betas, 
        warmup_epochs=cp.getint("TRAIN", "warmup_epochs"),
        lr_scheduler=cp.get("TRAIN", "lr_scheduler"),
        seed=cp.getint("TRAIN", "seed"),
        amp=_get_bool(cp.get("TRAIN", "amp")),
        device=cp.get("TRAIN", "device"),
        log_interval=cp.getint("TRAIN", "log_interval"),
    )
    save = SaveCfg(
        out_dir=cp.get("SAVE", "out_dir"),
        save_weights=_get_bool(cp.get("SAVE", "save_weights")),
        save_best=_get_bool(cp.get("SAVE", "save_best")),
        clean_workspace=(_get_bool(cp.get("SAVE", "clean_workspace"))
                         if cp.has_option("SAVE", "clean_workspace") else True),
    )
    return Cfg(data=data, model=model, train=train, save=save)
