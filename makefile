CC = gcc
Params = -c -O -std=c99

headers=main.h hashtables.h listas.h

executables: main.o hashtables.o listas.o cmdline.o
	$(CC) main.o hashtables.o listas.o cmdline.o -o cudaGen.exe 

main.o : main.c $(headers)
	$(CC) $(Params) main.c

hashtables.o: hashtables.c hashtables.h  
	$(CC) $(Params) hashtables.c

listas.o: listas.c listas.h
	$(CC) $(Params) listas.c
	
cmdline.o: cmdline.c cmdline.h
	$(CC) $(Params) cmdline.c


clean:
	rm -f *.o *.exe *.bak
