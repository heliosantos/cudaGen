
CC = nvcc

PROGRAM=testeee.exe

PROGRAM_OBJS=testeee.o 

.PHONY: clean all docs indent

${PROGRAM}: testeee.cu
	$(CC) $< -o $@

all: ${PROGRAM}

clean:
	rm -f *.o core.* *~ ${PROGRAM} *.bak 

docs: Doxyfile
	doxygen Doxyfile

Doxyfile:
	doxygen -g Doxyfile

indent:
	indent ${IFLAGS} *.cu *.h

