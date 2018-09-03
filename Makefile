EXE = fbsat
EXTRA =

all:
	@echo Specify some target

.PHONY: basic comb

basic: basic_tests-1 basic_tests-10 basic_tests-20 basic_tests-39
basic_tests-1:
	$(EXE) basic -i tests/simple/tests-1.gz -C 6 -K 2 -P 3 -N 14 $(EXTRA)
basic_tests-10:
	$(EXE) basic -i tests/simple/tests-10.gz -C 8 -K 4 -P 5 -N 25 $(EXTRA)
basic_tests-20:
	$(EXE) basic -i tests/simple/tests-20.gz -C 8 -K 4 -P 5 -N 25 $(EXTRA)
basic_tests-39:
	$(EXE) basic -i tests/simple/tests-39.gz -C 8 -K 4 -P 5 -N 25 $(EXTRA)

comb: comb_tests-1 comb_tests-10 comb_tests-20 comb_tests-39
comb_tests-1:
	$(EXE) combined -i tests/simple/tests-1.gz -C 6 -K 2 -P 3 -N 14 $(EXTRA)
comb_tests-10:
	$(EXE) combined -i tests/simple/tests-10.gz -C 8 -K 4 -P 5 -N 25 $(EXTRA)
comb_tests-20:
	$(EXE) combined -i tests/simple/tests-20.gz -C 8 -K 4 -P 5 -N 25 $(EXTRA)
comb_tests-39:
	$(EXE) combined -i tests/simple/tests-39.gz -C 8 -K 4 -P 5 -N 25 $(EXTRA)

tests-1_all_incremental: tests-1_full-min_incremental tests-1_partial-min_incremental tests-1_complete-min_incremental tests-1_minimize_incremental
tests-1_full-min_incremental:
	$(EXE) -i tests/simple/tests-1.gz --sat-solver incremental-cryptominisat --incremental full-min | tee logs/log$(shell date +%y%m%d)_$@
tests-1_partial-min_incremental:
	$(EXE) -i tests/simple/tests-1.gz --sat-solver incremental-cryptominisat --incremental partial-min | tee logs/log$(shell date +%y%m%d)_$@
tests-1_complete-min_incremental:
	$(EXE) -i tests/simple/tests-1.gz --sat-solver incremental-cryptominisat --incremental complete-min | tee logs/log$(shell date +%y%m%d)_$@
tests-1_minimize_incremental:
	$(EXE) -i tests/simple/tests-1.gz --sat-solver incremental-cryptominisat --incremental minimize --automaton out/minimal_partial_tests-1_C6_K6_T8.pkl | tee logs/log$(shell date +%y%m%d)_$@

tests-1_all: tests-1_full-min tests-1_partial-min tests-1_complete-min tests-1_minimize
tests-1_full-min:
	$(EXE) -i tests/simple/tests-1.gz --sat-solver cryptominisat5 full-min | tee logs/log$(shell date +%y%m%d)_$@
tests-1_partial-min:
	$(EXE) -i tests/simple/tests-1.gz --sat-solver cryptominisat5 partial-min | tee logs/log$(shell date +%y%m%d)_$@
tests-1_complete-min:
	$(EXE) -i tests/simple/tests-1.gz --sat-solver cryptominisat5 complete-min | tee logs/log$(shell date +%y%m%d)_$@
tests-1_minimize:
	$(EXE) -i tests/simple/tests-1.gz --sat-solver cryptominisat5 minimize --automaton out/minimal_partial_tests-1_C6_K6_T8.pkl | tee logs/log$(shell date +%y%m%d)_$@


tests-39_all_incremental: tests-39_full-min_incremental tests-39_partial-min_incremental tests-39_complete-min_incremental tests-39_minimize_incremental
tests-39_full-min_incremental:
	$(EXE) -i tests/simple/tests-39.gz --sat-solver incremental-cryptominisat --incremental full-min | tee logs/log$(shell date +%y%m%d)_$@
tests-39_partial-min_incremental:
	$(EXE) -i tests/simple/tests-39.gz --sat-solver incremental-cryptominisat --incremental partial-min | tee logs/log$(shell date +%y%m%d)_$@
tests-39_complete-min_incremental:
	$(EXE) -i tests/simple/tests-39.gz --sat-solver incremental-cryptominisat --incremental complete-min | tee logs/log$(shell date +%y%m%d)_$@
tests-39_minimize_incremental:
	$(EXE) -i tests/simple/tests-39.gz --sat-solver incremental-cryptominisat --incremental minimize --automaton out/minimal_partial_tests-39_C8_K8_T15.pkl | tee logs/log$(shell date +%y%m%d)_$@

tests-39_all: tests-39_full-min tests-39_partial-min tests-39_complete-min tests-39_minimize
tests-39_full-min:
	$(EXE) -i tests/simple/tests-39.gz --sat-solver cryptominisat5 full-min | tee logs/log$(shell date +%y%m%d)_$@
tests-39_partial-min:
	$(EXE) -i tests/simple/tests-39.gz --sat-solver cryptominisat5 partial-min | tee logs/log$(shell date +%y%m%d)_$@
tests-39_complete-min:
	$(EXE) -i tests/simple/tests-39.gz --sat-solver cryptominisat5 complete-min | tee logs/log$(shell date +%y%m%d)_$@
tests-39_minimize:
	$(EXE) -i tests/simple/tests-39.gz --sat-solver cryptominisat5 minimize --automaton out/minimal_partial_tests-39_C8_K8_T15.pkl | tee logs/log$(shell date +%y%m%d)_$@
