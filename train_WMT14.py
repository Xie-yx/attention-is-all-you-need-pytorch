import argparse
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from transformer import Transformer


def train(args):
    # 加载数据
    dataset = load_dataset('wmt14', 'de-en')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def encode(examples):
        return tokenizer(examples['translation']['en'], truncation=True, padding='max_length', max_length=512)

    def encode_target(examples):
        return tokenizer(examples['translation']['de'], truncation=True, padding='max_length', max_length=512)

    train_data = dataset['train'].map(encode, batched=True)
    val_data = dataset['test'].map(encode, batched=True)

    # 设置数据加载器
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    # 获取词汇大小
    n_src_vocab = tokenizer.vocab_size
    n_trg_vocab = tokenizer.vocab_size
    src_pad_idx = tokenizer.pad_token_id
    trg_pad_idx = tokenizer.pad_token_id

    # 初始化模型
    model = Transformer(
        n_src_vocab=n_src_vocab, n_trg_vocab=n_trg_vocab,
        src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx,
        d_word_vec=512, d_model=512, d_inner=2048,
        n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1
    ).to(args.device)

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(step ** -0.5,
                                                                                        step * args.warmup_steps ** -1.5))

    # 训练过程
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for step, batch in tqdm(enumerate(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}'):
            src_seq = batch['input_ids'].to(args.device)
            trg_seq = batch['labels'].to(args.device)

            optimizer.zero_grad()
            outputs = model(src_seq, trg_seq)
            loss = torch.nn.functional.cross_entropy(outputs, trg_seq.view(-1), ignore_index=trg_pad_idx)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss / len(train_loader)}')

    # 保存模型
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-4)  # Learning rate for Transformer base model
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--warmup_steps', type=int, default=4000)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
