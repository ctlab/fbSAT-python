EXE = fbsat
EXTRA = --sat-solver incremental-cryptominisat --incremental

all:
	@echo Specify some target

tests-1: tests-1_basic tests-1_basic-min tests-1_extended tests-1_extended-min tests-1_extended-min-ub tests-1_extended-min-ub-w2
tests-1_basic:
	$(EXE) -i data/scenarios/simple/tests-1.gz -o out/tests-1/basic -m basic -C 6 $(EXTRA)
tests-1_basic-min:
	$(EXE) -i data/scenarios/simple/tests-1.gz -o out/tests-1/basic-min -m basic-min $(EXTRA)
tests-1_extended:
	$(EXE) -i data/scenarios/simple/tests-1.gz -o out/tests-1/extended -m extended -C 6 -P 5 $(EXTRA)
tests-1_extended-min:
	$(EXE) -i data/scenarios/simple/tests-1.gz -o out/tests-1/extended-min -m extended-min -P 5 $(EXTRA)
tests-1_extended-min-ub:
	$(EXE) -i data/scenarios/simple/tests-1.gz -o out/tests-1/extended-min-ub -m extended-min-ub $(EXTRA)
tests-1_extended-min-ub-w2:
	$(EXE) -i data/scenarios/simple/tests-1.gz -o out/tests-1/extended-min-ub-w2 -m extended-min-ub -w 2 $(EXTRA)

tests-10: tests-10_basic tests-10_basic-min tests-10_extended tests-10_extended-min tests-10_extended-min-ub tests-10_extended-min-ub-w2
tests-10_basic:
	$(EXE) -i data/scenarios/simple/tests-10.gz -o out/tests-10/basic -m basic -C 8 $(EXTRA)
tests-10_basic-min:
	$(EXE) -i data/scenarios/simple/tests-10.gz -o out/tests-10/basic-min -m basic-min $(EXTRA)
tests-10_extended:
	$(EXE) -i data/scenarios/simple/tests-10.gz -o out/tests-10/extended -m extended -C 8 -P 5 $(EXTRA)
tests-10_extended-min:
	$(EXE) -i data/scenarios/simple/tests-10.gz -o out/tests-10/extended-min -m extended-min -P 5 $(EXTRA)
tests-10_extended-min-ub:
	$(EXE) -i data/scenarios/simple/tests-10.gz -o out/tests-10/extended-min-ub -m extended-min-ub $(EXTRA)
tests-10_extended-min-ub-w2:
	$(EXE) -i data/scenarios/simple/tests-10.gz -o out/tests-10/extended-min-ub-w2 -m extended-min-ub -w 2 $(EXTRA)

tests-39: tests-39_basic tests-39_basic-min tests-39_extended tests-39_extended-min tests-39_extended-min-ub tests-39_extended-min-ub-w2
tests-39_basic:
	$(EXE) -i data/scenarios/simple/tests-39.gz -o out/tests-39/basic -m basic -C 8 $(EXTRA)
tests-39_basic-min:
	$(EXE) -i data/scenarios/simple/tests-39.gz -o out/tests-39/basic-min -m basic-min $(EXTRA)
tests-39_extended:
	$(EXE) -i data/scenarios/simple/tests-39.gz -o out/tests-39/extended -m extended -C 8 -P 5 $(EXTRA)
tests-39_extended-min:
	$(EXE) -i data/scenarios/simple/tests-39.gz -o out/tests-39/extended-min -m extended-min -P 5 $(EXTRA)
tests-39_extended-min-ub:
	$(EXE) -i data/scenarios/simple/tests-39.gz -o out/tests-39/extended-min-ub -m extended-min-ub $(EXTRA)
tests-39_extended-min-ub-w2:
	$(EXE) -i data/scenarios/simple/tests-39.gz -o out/tests-39/extended-min-ub-w2 -m extended-min-ub -w 2 $(EXTRA)

tests-59: tests-59_basic tests-59_basic-min tests-59_extended tests-59_extended-min tests-59_extended-min-ub tests-59_extended-min-ub-w2
tests-59_basic:
	$(EXE) -i data/scenarios/hard/tests-10-60.gz -o out/tests-59/basic -m basic -C 8 $(EXTRA)
tests-59_basic-min:
	$(EXE) -i data/scenarios/hard/tests-10-60.gz -o out/tests-59/basic-min -m basic-min $(EXTRA)
tests-59_extended:
	$(EXE) -i data/scenarios/hard/tests-10-60.gz -o out/tests-59/extended -m extended -C 8 -P 5 $(EXTRA)
tests-59_extended-min:
	$(EXE) -i data/scenarios/hard/tests-10-60.gz -o out/tests-59/extended-min -m extended-min -P 5 $(EXTRA)
tests-59_extended-min-ub:
	$(EXE) -i data/scenarios/hard/tests-10-60.gz -o out/tests-59/extended-min-ub -m extended-min-ub $(EXTRA)
tests-59_extended-min-ub-w2:
	$(EXE) -i data/scenarios/hard/tests-10-60.gz -o out/tests-59/extended-min-ub-w2 -m extended-min-ub -w 2 $(EXTRA)
