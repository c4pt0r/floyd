import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--forward-output")
    args = parser.parse_args()

    model_dir = Path(args.model)
    sys.path.insert(0, str(model_dir / "encoding"))
    from encoding_dsv4 import encode_messages

    conversations = [
        [{"role": "user", "content": "hello"}],
        [{"role": "user", "content": "你好，世界！"}],
        [{"role": "user", "content": "Numbers 1 12 123 1234; punctuation?!"}],
        [{"role": "user", "content": "first line\nsecond line\n\nlast"}],
        [{"role": "user", "content": "!!hello --world ++test"}],
        [{"role": "user", "content": "Cafe\u0301 deja\u0300 vu"}],
        [{"role": "user", "content": "中文ABC日本語xyz"}],
        [{"role": "user", "content": "snake_case::HTTPServer"}],
        [{"role": "user", "content": "a  \t b\r\n\r\n c"}],
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2=4."},
            {"role": "user", "content": "Now multiply by 3."},
        ],
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    cases = []
    for messages in conversations:
        prompt = encode_messages(messages, thinking_mode="chat")
        ids = tokenizer.encode(prompt)
        cases.append({
            "messages": messages,
            "prompt": prompt,
            "ids": ids,
            "decoded": tokenizer.decode(ids),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2))
    if args.forward_output:
        from make_deepseek_v4_forward_oracle import write_base_forward_oracle
        write_base_forward_oracle(model_dir, args.forward_output, cases[0]["ids"])


if __name__ == "__main__":
    main()
