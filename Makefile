.POSIX:

install:
	poetry install

download:
	poetry run download-emnist
	poetry run download-nltk
	poetry run download-iam

generate:
	poetry run generate-emnist-lines 
	poetry run generate-iam-lines
	poetry run generate-iam-paragraphs
	poetry run generate-iam-synthetic-paragraphs
	poetry run extract-iam-text --use_words --save_text train.txt --save_tokens letters.txt
	poetry run make-wordpieces --output_prefix iamdb_1kwp --text_file train.txt --num_pieces 1000

.PHONY: download generate
