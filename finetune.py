import os
import json
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from config import settings

def load_conversation_data(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def prepare_dataset(entries: List[dict], tokenizer, max_length: int = 1024):
    inputs, labels = [], []
    for conv in entries:
        prompt = f"User: {conv['user']}\nBot: {conv['bot']}\n"
        tokenized = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = tokenized.input_ids.squeeze()
        inputs.append(input_ids)
        labels.append(input_ids.clone())
    return inputs, labels

def fine_tune(model_name: str, data_path: str, output_dir: str, epochs: int = 3, batch_size: int = 4):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    model = get_peft_model(base_model, peft_config)
    data = load_conversation_data(data_path)
    inputs, labels = prepare_dataset(data, tokenizer)
    class DatasetConv(torch.utils.data.Dataset):
        def __len__(self):
            return len(inputs)
        def __getitem__(self, idx):
            return {"input_ids": inputs[idx], "labels": labels[idx]}
    dataset = DatasetConv()
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=10,
        save_total_limit=2,
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: {
            "input_ids": torch.nn.utils.rnn.pad_sequence([f["input_ids"] for f in data], batch_first=True),
            "labels": torch.nn.utils.rnn.pad_sequence([f["labels"] for f in data], batch_first=True),
        },
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
