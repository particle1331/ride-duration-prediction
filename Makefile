clean: clean-test clean-build
	find . | grep ".vscode" | xargs rm -rf
	find . | grep ".DS_Store" | xargs rm
	find . | grep "__pycache__" | xargs rm -rf
	find . | grep ".ipynb_checkpoints" | xargs rm -rf

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . | grep ".egg" | xargs rm -rf
	find . | grep ".egg-info" | xargs rm -rf

clean-test:
	rm -f .coverage
	rm -rf htmlcov/
	find . | grep ".pytest_cache" | xargs rm -rf

lint:
	pylint --recursive=y ride_duration

test:
	pytest -W ignore -vv ride_duration

format:
	isort .
	black .

coverage:
	coverage run -m pytest -vv ride_duration
	coverage report -m
