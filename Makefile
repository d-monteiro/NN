CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -Iinclude
SRC = $(wildcard src/*.cpp)
OBJ = $(SRC:src/%.cpp=build/%.o)
TARGET = build/NN

# Test files
#TEST_SRC = tests/test_main.cpp
#TEST_OBJ = $(TEST_SRC:tests/%.cpp=build/tests/%.o)
#TEST_TARGET = build/tetris_test

# Exclude main.cpp when building tests
#TEST_SRC_DEPS = $(filter-out src/main.cpp, $(SRC))
#TEST_OBJ_DEPS = $(TEST_SRC_DEPS:src/%.cpp=build/%.o)

all: clean $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@

build/%.o: src/%.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Test targets
test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJ) $(TEST_OBJ_DEPS)
	$(CXX) $(TEST_OBJ) $(TEST_OBJ_DEPS) -o $@ $(LDFLAGS)

build/tests/%.o: tests/%.cpp
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) -c $< -o $@

check: test

clean:
	rm -rf build/*

clear:
	rm -rf build/*

run:
	./build/NN

.PHONY: all clean test check run