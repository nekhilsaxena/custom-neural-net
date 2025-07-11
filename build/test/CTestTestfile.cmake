# CMake generated Testfile for 
# Source directory: C:/Nekhil/Projects/RobotArm/NeuralNetwork/test
# Build directory: C:/Nekhil/Projects/RobotArm/NeuralNetwork/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(NeuralNetworkTest "C:/Nekhil/Projects/RobotArm/NeuralNetwork/build/test/Debug/NeuralNetworkTest.exe")
  set_tests_properties(NeuralNetworkTest PROPERTIES  _BACKTRACE_TRIPLES "C:/Nekhil/Projects/RobotArm/NeuralNetwork/test/CMakeLists.txt;14;add_test;C:/Nekhil/Projects/RobotArm/NeuralNetwork/test/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(NeuralNetworkTest "C:/Nekhil/Projects/RobotArm/NeuralNetwork/build/test/Release/NeuralNetworkTest.exe")
  set_tests_properties(NeuralNetworkTest PROPERTIES  _BACKTRACE_TRIPLES "C:/Nekhil/Projects/RobotArm/NeuralNetwork/test/CMakeLists.txt;14;add_test;C:/Nekhil/Projects/RobotArm/NeuralNetwork/test/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(NeuralNetworkTest "C:/Nekhil/Projects/RobotArm/NeuralNetwork/build/test/MinSizeRel/NeuralNetworkTest.exe")
  set_tests_properties(NeuralNetworkTest PROPERTIES  _BACKTRACE_TRIPLES "C:/Nekhil/Projects/RobotArm/NeuralNetwork/test/CMakeLists.txt;14;add_test;C:/Nekhil/Projects/RobotArm/NeuralNetwork/test/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(NeuralNetworkTest "C:/Nekhil/Projects/RobotArm/NeuralNetwork/build/test/RelWithDebInfo/NeuralNetworkTest.exe")
  set_tests_properties(NeuralNetworkTest PROPERTIES  _BACKTRACE_TRIPLES "C:/Nekhil/Projects/RobotArm/NeuralNetwork/test/CMakeLists.txt;14;add_test;C:/Nekhil/Projects/RobotArm/NeuralNetwork/test/CMakeLists.txt;0;")
else()
  add_test(NeuralNetworkTest NOT_AVAILABLE)
endif()
