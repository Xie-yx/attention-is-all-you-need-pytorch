import argparse
import torch
from transformers import T5Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from transformer import Transformer


def evaluate(args):
    # 加载数据
    dataset = load_dataset('wmt14', 'de-en')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def encode(examples):
        return tokenizer(examples['translation']['en'], truncation=True, padding='max_length', max_length=512)

    def encode_target(examples):
        return tokenizer(examples['translation']['de'], truncation=True, padding='max_length', max_length=512)

    val_data = dataset['test'].map(encode, batched=True)

    # 设置数据加载器
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    # 加载模型
    model = Transformer.from_pretrained(args.model_dir).to(args.device)
    model.eval()  # 设置为评估模式

    total_bleu = 0
    total_examples = 0

    # 评估过程
    for batch in tqdm(val_loader, desc='Evaluating'):
        src_seq = batch['input_ids'].to(args.device)
        trg_seq = batch['labels'].to(args.device)

        # 模型预测
        with torch.no_grad():
            outputs = model(src_seq, trg_seq)

        # 计算BLEU分数（这里只是一个简单的示范，通常需要外部工具计算）
        predicted = outputs.argmax(dim=-1)
        bleu_score = calculate_bleu(predicted, trg_seq)
        total_bleu += bleu_score
        total_examples += 1

    avg_bleu = total_bleu / total_examples
    print(f'Average BLEU score: {avg_bleu}')


def calculate_bleu(predicted, target):
    # 一个简单的BLEU计算示例，你可以使用具体的库（如nltk）来计算BLEU
    correct = (predicted == target).float().sum().item()
    total = target.size(0) * target.size(1)  # batch_size * sequence_length
    bleu_score = correct / total
    return bleu_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
