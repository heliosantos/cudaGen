#flags para o compilador
CFLAGS = -Wall -W -g -Wmissing-prototypes

# BIBLIOTECAS
LIBS = #

#++++++++++++++++  ficheiros objectos  +++++++++++++++++++++
OBJS = ignore.o

# nome do executavel 
PROGRAM = ignore.exe
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 

# make principal
all: ${PROGRAM}

# compilar com depuração
dbug: CFLAGS += -D SHOW_DEBUG 
dbug: ${PROGRAM}

docs: 
	doxygen Doxyfile

	

${PROGRAM}: ${OBJS}
	${CC} -o $@ ${OBJS} ${LIBS}


#++++ Lista de dependências dos ficheiros código fonte +++++
ignore.o: ignore.c ignore.h
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Indica como transformar um ficheiro .c num ficheiro .o
.c.o:
	${CC} ${CFLAGS} -c $<
	
# remove ficheiros sem interesse
clean:
	rm -f *.o core.* *~ ${PROGRAM} *.bak 
	
