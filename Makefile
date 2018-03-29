PY = python
SCRIPY = main.py
EXTRA =

all:
	@echo Specify some target

tests-1:
	${PY} ${SCRIPY} -t tests/simple/tests-1.gz --basename /tmp/cnf -C 6 -K 2 -P 3 -N 14 ${EXTRA}
tests-10:
	${PY} ${SCRIPY} -t tests/simple/tests-10.gz --basename /tmp/cnf -C 8 -K 4 -P 5 -N 25 ${EXTRA}
tests-20:
	${PY} ${SCRIPY} -t tests/simple/tests-20.gz --basename /tmp/cnf -C 8 -K 4 -P 5 -N 25 ${EXTRA}
tests-39:
	${PY} ${SCRIPY} -t tests/simple/tests-39.gz --basename /tmp/cnf -C 8 -K 4 -P 5 -N 25 ${EXTRA}

.PHONY: tests-1
.PHONY: tests-10
.PHONY: tests-20
.PHONY: tests-39
