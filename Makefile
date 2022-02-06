.POSIX:

check:
	nox

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

.PHONY: install download generate
