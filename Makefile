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

graph:
	poetry run build-transitions --prune 0 10 --blank optional --self_loops --save_path 1kwp_prune_0_10_optblank.bin

.PHONY: install download generate graph
