all:
	@echo Specify some target

.PHONY: tests-1
tests-1:
	# N = 14
	python main.py -t tests/simple/tests-1.gz --basename /tmp/cnf --predicate-names real_predicate-names -C 6 -K 2 -P 3 --min

.PHONY: tests-10
tests-10:
	# N <= 25
	python main.py -t tests/simple/tests-10.gz --basename /tmp/cnf --predicate-names real_predicate-names -C 8 -K 4 -P 5 --min

.PHONY: tests-39
tests-39:
	# N <= 25
	python main.py -t tests/simple/tests-39.gz --basename /tmp/cnf --predicate-names real_predicate-names -C 8 -K 4 -P 5 --min
