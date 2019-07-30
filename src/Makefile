requirements:
	pipenv lock -r | tail -n +2 > requirements.txt
	pipenv lock -r -d | tail -n +2 > requirements-dev.txt

format:
	pipenv run yapf -i -r .

check-format:
	pipenv run flake8

bandit:
	pipenv run bandit -r .

.PHONY: requirements format check-format bandit
