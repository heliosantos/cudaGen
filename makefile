#flags para o compilador
CFLAGS = -Wall -W -g -Wmissing-prototypes

# BIBLIOTECAS
LIBS = #

# third party dir
Third = 3rdParty

#++++++++++++++++  ficheiros objectos  +++++++++++++++++++++
OBJS = main.o cmdline.o dirutils.o hashtables.o listas.o debug.o templateGen.o

# nome do executavel 
PROGRAM = cudagen.exe
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
main.o: main.c main.h cmdline.h dirutils.h ${Third}/hashtables.h ${Third}/listas.h ${Third}/debug.h templateGen.h
dirutils.o: dirutils.c dirutils.h
cmdline.o: cmdline.c cmdline.h
hashtables.o: ${Third}/hashtables.c ${Third}/hashtables.h
listas.o: ${Third}/listas.c ${Third}/listas.h
templateGen.o: templateGen.c templateGen.h
debug.o: ${Third}/debug.c ${Third}/debug.h
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Indica como transformar um ficheiro .c num ficheiro .o
.c.o:
	${CC} ${CFLAGS} -c $<
	

listas.o: ${Third}/listas.c ${Third}/listas.h
	${CC} ${CFLAGS} -c ${Third}/listas.c
hashtables.o: ${Third}/hashtables.c ${Third}/hashtables.h
	${CC} ${CFLAGS} -c ${Third}/hashtables.c
debug.o: ${Third}/debug.c ${Third}/debug.h
	${CC} ${CFLAGS} -c ${Third}/debug.c

# remove ficheiros sem interesse
clean:
	rm -f *.o core.* *~ ${PROGRAM} *.bak 
	
