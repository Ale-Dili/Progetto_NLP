import json
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
import os


class CDA_Debiaser:

    def __init__(self, model, model_name ,tokenizer,device='mps'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.OUTPUT_PATH = os.path.join('./debiased_model',model_name+'_cda_debiased')

        self.model.eval()
        self.model.to(device)

        self.SEED = 44

        self.base_texts = []
        with open('data/cda/base_texts.json') as f:
            self.base_texts = json.load(f)

    def _tokenize_function(self,examples):
        return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    def debias(self):
        dataset = Dataset.from_dict({"text": self.base_texts})

        tokenized = dataset.map(self._tokenize_function, batched=True, remove_columns=["text"])

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )


        training_args = TrainingArguments(
            output_dir=".",  
            num_train_epochs=4,
            per_device_train_batch_size=8,
            save_steps=500,
            save_total_limit=2,
            save_strategy='no',
            logging_strategy='no',
            logging_dir=None,
            report_to=[] 
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator
        )

        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        trainer.train()

        trainer.save_model(self.OUTPUT_PATH)
        print("Model succesfully saved in: ", self.OUTPUT_PATH)
        

