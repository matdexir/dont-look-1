import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    batch_size = args.batch_size
    device = args.device

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len, mode="train")
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(
        datasets[TRAIN],
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=datasets[TRAIN].collate_fn,
    )
    dev_loader = DataLoader(
        datasets[DEV],
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=datasets[DEV].collate_fn,
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_class=args.num_class,
        # recurrent=args.recurrent,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        # prefix=args.prefix,
    )

    # TODO: init optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0)
    model.to(device=args.device)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        best_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        # h = model.init_hidden(batch_size, device)
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data["text"], data["intent"]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out, _ = model(inputs, None)
            loss = nn.CrossEntropyLoss(out, labels)
            _, train_pred = torch.max(out, 1)
            loss.backward()
            optimizer.step()

            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += loss.item()

        with torch.no_grad():
            # h = model.init_hidden(batch_size, device)
            model.eval()
            for i, dev_data in enumerate(tqdm(dev_loader)):
                inputs, labels = dev_data["text"].to(device), dev_data["intent"].to(
                    device
                )
                out, _ = model(inputs, None)

                loss = nn.CrossEntropyLoss(out, labels)
                _, val_pred = torch.max(out, 1)
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += loss.item()

            print(
                f"Epoch {epoch + 1}: Train Acc: {train_acc / len(train_loader.dataset)}, Train Loss: {train_loss / len(train_loader)}, Val Acc: {val_acc / len(dev_loader.dataset)}, Val Loss: {val_loss / len(dev_loader)}"
            )
            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "./ckpt/intent/model.ckpt")
                print(f"Save model with acc {val_acc / len(dev_loader.dataset)}")
        # TODO: Evaluation loop - calculate accuracy and save model weights
    testing_data = None
    with open("./data/intent/test.json", "r") as fp:
        testing_data = json.load(fp)
    testing_dataset = SeqClsDataset(testing_data, vocab, intent2idx, args.max_len)
    # TODO: create DataLoader for test dataset
    tt_loader = torch.utils.data.DataLoader(
        testing_dataset,
        shuffle=False,
        batch_size=150,
        collate_fn=testing_dataset.collate_fn,
    )
    model.eval()
    ids = [d["id"] for d in testing_data]
    # load weights into model

    # TODO: predict dataset
    # preds = []
    ids = [d["id"] for d in testing_data]
    with open("./pred.intent.csv", "w") as fp:
        fp.write("id,intent\n")
        with torch.no_grad():
            for i, d in enumerate(tqdm(tt_loader)):
                out, _ = model(d["text"].to("cuda:0"), None)
                _, pred = torch.max(out, 1)
                for j, p in enumerate(pred):
                    fp.write(f"{ids[150*i+j]},{testing_dataset.idx2label(p.item())}\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
