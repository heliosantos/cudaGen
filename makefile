#flags para o compilador
CFLAGS = -Wall -W -g -Wmissing-prototypes

# BIBLIOTECAS
LIBS = #


#++++++++++++++++  ficheiros objectos  +++++++++++++++++++++
OBJS = main.o cmdline.o dirutils.o hashtables.o listas.o

# nome do executavel 
PROGRAM = cudagen
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
main.o: main.c main.h cmdline.h dirutils.h hashtables.h listas.h
dirutils.o: dirutils.c dirutils.h
cmdline.o: cmdline.c cmdline.h
hashtables.o: hashtables.c hashtables.h
listas.o: listas.c listas.h
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Indica como transformar um ficheiro .c num ficheiro .o
.c.o:
	${CC} ${CFLAGS} -c $<


# remove ficheiros sem interesse
clean:
	rm -f *.o core.* *~ ${PROGRAM} *.bak 
