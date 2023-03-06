import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Set device to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model for text summarization
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_model.to(device)
# Load BART tokenizer
bart_tok = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def summarize(sentences):
    # Define function for single text summarization
    def single_summarise(input_sent_ids):
        # Generate summary
        summary_ids = bart_model.generate(input_sent_ids, max_length=510, num_beams=4, early_stopping=True)
        # Decode summary back to text
        summary_text = bart_tok.decode(summary_ids[0], skip_special_tokens=True)
        # Return summarized text
        return summary_text

    # Define function for summarization of too large texts
    def large_summarize(input_ids):
        # Initialize result as an empty tensor
        summ = torch.tensor([], dtype=torch.long).to(device)
        # While there is more then 1024 ids in input
        while len(input_ids[0])>1024:
            # Get first 1023 ids and add SEP token at the end
            part = torch.cat((input_ids[0][:1023].unsqueeze(-2), torch.tensor([2]).unsqueeze(-2).to(device)), dim=-1)
            # Generate summary of this firs 1024 tokens
            sum_part = bart_model.generate(part, max_length=510, num_beams=4, early_stopping=True)
            # Add summary to the result
            summ = torch.cat((summ, sum_part), dim=-1)
            # Remove summurized ids and add [CLS] in the beggining
            input_ids = torch.cat((torch.tensor([0]).unsqueeze(-2).to(device), input_ids[0][1023:].unsqueeze(-2)), dim=-1)
        # After loop is done, get summary of the last ids
        sum_part = bart_model.generate(input_ids, max_length=510, num_beams=4, early_stopping=True)
        # And add it to the result
        summ = torch.cat((summ, sum_part), dim=-1)
        # Return the result
        return summ

    # For every sentence...
    for i in range(len(sentences)):
        if i%10000 == 0:
            print(f'Processed over {i} elements')

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = bart_tok.encode(sentences[i], return_tensors="pt").clone().detach().requires_grad_(False).to(device)

        # If sentence length is between 512 and 1024
        if 512<len(input_ids[0])<=1024:
            # Summarize sententce to max 510 tokens
            sentences[i] = single_summarise(input_ids)
        # If sentence is more then 1024 tokens
        elif len(input_ids[0])>1024:
            # While its more then 510 tokens
            while len(input_ids[0])>510:
                # Summurize it
                input_ids = large_summarize(input_ids)
            # After its summurized, decode it and update the sentence
            sentences[i] = bart_tok.decode(input_ids[0], skip_special_tokens=True)