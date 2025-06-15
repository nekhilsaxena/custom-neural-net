#pragma once
#include <vector>
#include <GLFW/glfw3.h>

class Visualization
{
public:
     void initialize();
     void shutdown();

     void beginFrame();
     void endFrame();

     void updateCost(float epoch, float cost);
     void render();

private:
     std::vector<float> epochHistory;
     std::vector<float> costHistory;
     bool hasPlottedOnce = false;
};

extern GLFWwindow *g_Window;
void setVisualizationWindow(GLFWwindow *window);
