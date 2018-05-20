EXE = fbsat
EXTRA =

all:
	@echo Specify some target

.PHONY: basic comb

basic: basic_tests-1 basic_tests-10 basic_tests-20 basic_tests-39
basic_tests-1:
	${EXE} basic -i tests/simple/tests-1.gz -C 6 -K 2 -P 3 -N 14 ${EXTRA}
basic_tests-10:
	${EXE} basic -i tests/simple/tests-10.gz -C 8 -K 4 -P 5 -N 25 ${EXTRA}
basic_tests-20:
	${EXE} basic -i tests/simple/tests-20.gz -C 8 -K 4 -P 5 -N 25 ${EXTRA}
basic_tests-39:
	${EXE} basic -i tests/simple/tests-39.gz -C 8 -K 4 -P 5 -N 25 ${EXTRA}

comb: comb_tests-1 comb_tests-10 comb_tests-20 comb_tests-39
comb_tests-1:
	${EXE} combined -i tests/simple/tests-1.gz -C 6 -K 2 -P 3 -N 14 ${EXTRA}
comb_tests-10:
	${EXE} combined -i tests/simple/tests-10.gz -C 8 -K 4 -P 5 -N 25 ${EXTRA}
comb_tests-20:
	${EXE} combined -i tests/simple/tests-20.gz -C 8 -K 4 -P 5 -N 25 ${EXTRA}
comb_tests-39:
	${EXE} combined -i tests/simple/tests-39.gz -C 8 -K 4 -P 5 -N 25 ${EXTRA}
