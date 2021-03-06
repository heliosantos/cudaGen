
CC = nvcc

PROGRAM=$!FILENAME!$.exe

PROGRAM_OBJS=$!FILENAME!$.o 

.PHONY: clean all docs indent

${PROGRAM}: $!FILENAME!$.cu
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

