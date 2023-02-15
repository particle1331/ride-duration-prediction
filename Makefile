.PHONY: clean-pyc clean-build clean

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".vscode" | xargs rm -rf

clean-test:
	rm -f .coverage
	rm -fr htmlcov/

lint:
	pylint --recursive=y ride_duration

test:
	pytest -W ignore ride_duration

format:
	black .
	isort .

coverage:
	coverage run -m pytest ride_duration
	coverage report -m
