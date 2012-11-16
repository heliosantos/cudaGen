
CC = nvcc

PROGRAM=ignoresss.exe

PROGRAM_OBJS=ignoresss.o 

.PHONY: clean all docs indent

${PROGRAM}: ignoresss.cu
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

