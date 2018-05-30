# Minimal Makefile to compile NaiBX for MLC experiments

CXX = g++ -std=c++11 -g -I.
CXXFLAGS = -O3  -Wall

EXE = naibx
SRC = include
TST = test
OBJ = managedata.o metrics.o

$(EXE):    $(OBJ)    main.o
	$(CXX) $(CXXFLAGS) $(OBJ)    main.o -o $(EXE)

managedata.o: $(SRC)/managedata.cpp
	$(CXX) $(CXXFLAGS) -c $(SRC)/managedata.cpp -o managedata.o
metrics.o:    $(SRC)/metrics.cpp managedata.o
	$(CXX) $(CXXFLAGS) -c $(SRC)/metrics.cpp -o metrics.o
bownaibx.o: $(OBJ) $(SRC)/bownaibx.hpp
	$(CXX) $(CXXFLAGS) -c $(SRC)/bownaibx.hpp  -o bownaibx.o
naibx.o: $(OBJ) $(SRC)/naibx.hpp
	$(CXX) $(CXXFLAGS) -c $(SRC)/naibx.hpp -o naibx.o
main.o:  $(OBJ) naibx.o bownaibx.o $(TST)/main.cpp
	$(CXX) $(CXXFLAGS) -c $(TST)/main.cpp  -o main.o


.PHONY : clean cleanup
clean :
	rm $(OBJ) naibx.o bownaibx.o main.o naibx
cleanup :
	rm *.o
